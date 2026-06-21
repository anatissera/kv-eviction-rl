"""
Reusable evaluation core for online KV-cache eviction.

These functions are shared by the offline CLI evaluator (`scripts/eval.py`) and the
training-time monitoring probe (`kv_gym.probe.EvalProbeCallback`).  Keeping the single
online decode loop (`run_online_episode`) in one place means the probe measures exactly
what the CLI eval measures — same eviction semantics, same RoPE/position handling.

All strategies run ONLINE: at each decode step, if cache_size > budget, one token is
evicted per layer before the next token is generated.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from kv_gym.features import build_obs
from kv_gym.vendor.answer_extraction_gsm8k import flexible_extract


# ── Cache utilities ───────────────────────────────────────────────────────────

def _get_cache_size(past_kv) -> int:
    if hasattr(past_kv, "layers"):
        return past_kv.layers[0].keys.shape[2]
    return past_kv.key_cache[0].shape[2]


def _extract_all_kv(past_kv, L: int) -> tuple[Tensor, Tensor]:
    """Return K, V stacked over layers: [L, H, seq, D] on CPU float32."""
    K_list, V_list = [], []
    for l in range(L):
        if hasattr(past_kv, "layers"):
            k, v = past_kv.layers[l].keys, past_kv.layers[l].values
        else:
            k, v = past_kv.key_cache[l], past_kv.value_cache[l]
        K_list.append(k.squeeze(0).float().cpu())
        V_list.append(v.squeeze(0).float().cpu())
    return torch.stack(K_list), torch.stack(V_list)


def _evict_per_layer(past_kv, evict_slots: list[int]) -> None:
    """Remove one slot per layer from the live DynamicCache."""
    if hasattr(past_kv, "layers"):
        for l, s in enumerate(evict_slots):
            seq = past_kv.layers[l].keys.shape[2]
            keep = torch.tensor([i for i in range(seq) if i != s],
                                device=past_kv.layers[l].keys.device)
            past_kv.layers[l].keys   = past_kv.layers[l].keys.index_select(2, keep)
            past_kv.layers[l].values = past_kv.layers[l].values.index_select(2, keep)
    else:
        for l, s in enumerate(evict_slots):
            seq = past_kv.key_cache[l].shape[2]
            keep = torch.tensor([i for i in range(seq) if i != s],
                                device=past_kv.key_cache[l].device)
            past_kv.key_cache[l]   = past_kv.key_cache[l].index_select(2, keep)
            past_kv.value_cache[l] = past_kv.value_cache[l].index_select(2, keep)


# ── Position tracker ──────────────────────────────────────────────────────────

class PositionTracker:
    """Tracks original sequence position of each cache slot, per layer.

    Necessary for the attention oracle: after evictions, cache slots no longer
    correspond 1-to-1 to original positions, so we can't look up importance
    directly by slot index.
    """

    def __init__(self, L: int, T: int):
        self._pos: list[list[int]] = [list(range(T)) for _ in range(L)]

    def evict(self, layer: int, slot: int) -> None:
        self._pos[layer].pop(slot)

    def append_all(self, original_pos: int) -> None:
        """New generated token appended to every layer at the given position."""
        for p in self._pos:
            p.append(original_pos)

    def original_pos(self, layer: int, slot: int) -> int:
        return self._pos[layer][slot]

    def size(self, layer: int = 0) -> int:
        return len(self._pos[layer])


# ── Recency + sink protection ──────────────────────────────────────────────────

def valid_action_mask(
    cache_size: int, max_len: int, n_sinks: int = 0, n_recent: int = 0
) -> np.ndarray:
    """[max_len] bool mask of evictable slots: protects the first `n_sinks` (attention
    sinks) and the last `n_recent` (recency window) cache slots.

    The live cache stays sorted by original position ascending (evictions compact
    preserving order; new tokens append at the end), so slot 0..n_sinks-1 are the
    oldest/sink tokens and slot cache_size-n_recent..cache_size-1 are the most recent.

    Guard: if the window would cover the whole cache, fall back to all-evictable so
    MaskablePPO always has ≥1 valid action.
    """
    mask = np.zeros(max_len, dtype=bool)
    lo = min(n_sinks, max(0, cache_size - 1))
    hi = max(lo, cache_size - n_recent)
    if hi <= lo:                      # window covers everything → relax to all slots
        mask[:cache_size] = True
    else:
        mask[lo:hi] = True
    return mask


def _valid_slots(cache_size: int, n_sinks: int, n_recent: int) -> list[int]:
    """Indices of evictable slots given the recency+sink window (same policy as
    valid_action_mask), for baselines that pick among candidates."""
    m = valid_action_mask(cache_size, cache_size, n_sinks, n_recent)
    return [i for i in range(cache_size) if m[i]]


# ── Strategy factories ────────────────────────────────────────────────────────

def make_learned_evict_fn(policy, L: int, max_len: int, n_sinks: int = 0, n_recent: int = 0):
    """PPO policy: observe current K/V and predict one eviction slot per layer.

    The action mask protects the first `n_sinks` and last `n_recent` cache slots —
    must match the env's action_masks() used during training.
    """
    def evict(K: Tensor, V: Tensor, tracker: PositionTracker) -> list[int]:
        cache_size = K.shape[2]
        obs   = build_obs(K, V, cache_size, max_len)          # [L, max_len, feat]
        row   = valid_action_mask(cache_size, max_len, n_sinks, n_recent)
        masks = np.broadcast_to(row, (L, max_len)).copy()
        actions, _ = policy.predict(obs, action_masks=masks, deterministic=True)
        return [int(a) for a in actions]
    return evict


def make_streaming_evict_fn(n_sinks: int, L: int):
    """StreamingLLM: evict the oldest non-sink slot (always slot n_sinks)."""
    def evict(K: Tensor, V: Tensor, tracker: PositionTracker) -> list[int]:
        return [n_sinks] * L
    return evict


def make_attention_evict_fn(per_layer_imp: Tensor, L: int, n_sinks: int = 0, n_recent: int = 0):
    """Attention oracle: evict the valid slot whose original position has least future
    attention (restricted to the same recency+sink window as the learned policy)."""
    T_imp = per_layer_imp.shape[1]

    def evict(K: Tensor, V: Tensor, tracker: PositionTracker) -> list[int]:
        slots = []
        for l in range(L):
            n = tracker.size(l)
            cand = _valid_slots(n, n_sinks, n_recent)
            imps = [
                per_layer_imp[l, tracker.original_pos(l, s)].item()
                if tracker.original_pos(l, s) < T_imp else 0.0
                for s in cand
            ]
            slots.append(cand[int(np.argmin(imps))])
        return slots

    return evict


def make_kv_norm_evict_fn(L: int, n_sinks: int = 0, n_recent: int = 0):
    """KV-norm: per layer, evict the valid slot with the lowest current ||K||+||V|| norm."""
    def evict(K: Tensor, V: Tensor, tracker: PositionTracker) -> list[int]:
        # K, V: [L, H, seq, D] — mean over heads
        norms = (K.norm(dim=-1) + V.norm(dim=-1)).mean(dim=1)  # [L, seq]
        cand = _valid_slots(K.shape[2], n_sinks, n_recent)
        cand_t = torch.tensor(cand)
        return [cand[int(norms[l, cand_t].argmin().item())] for l in range(L)]
    return evict


def make_random_evict_fn(L: int, rng, n_sinks: int = 0, n_recent: int = 0):
    """Random: uniformly random valid slot per layer (respects the window)."""
    def evict(K: Tensor, V: Tensor, tracker: PositionTracker) -> list[int]:
        cand = _valid_slots(K.shape[2], n_sinks, n_recent)
        return [cand[int(rng.integers(0, len(cand)))] for _ in range(L)]
    return evict


# ── Online episode runner ─────────────────────────────────────────────────────

@torch.no_grad()
def run_online_episode(
    model,
    tokenizer,
    input_ids:      Tensor,
    budget:         int,
    max_new_tokens: int,
    device:         torch.device,
    evict_fn,
    per_layer_imp:  Tensor | None = None,  # [L, T_prompt] for correlation tracking
) -> tuple[str, list[float], list[tuple[int, int]]]:
    """Run one full online episode.

    At each decode step the cache grows by one token.  When cache_size > budget
    the evict_fn removes one token per layer before decoding continues.

    Returns:
        text         — the generated continuation.
        correlation  — attention-rank percentile of each eviction decision (empty
                       if per_layer_imp is None).
        evicted      — list of (original_position, true_position_at_eviction) for every
                       per-layer eviction decision. true_position is the absolute
                       sequence position being generated at that step, so
                       original_position / true_position ∈ [0, 1) is a recency score
                       (0 = oldest token evicted, ~1 = most-recent evicted).
    """
    L   = model.config.num_hidden_layers
    T   = input_ids.shape[1]
    eos = tokenizer.eos_token_id

    out      = model(input_ids=input_ids.to(device), use_cache=True)
    past_kv  = out.past_key_values
    next_tok = int(out.logits[0, -1].argmax())
    true_pos = T
    generated:   list[int] = []
    correlation: list[float] = []
    evicted:     list[tuple[int, int]] = []
    tracker = PositionTracker(L, T)

    for _ in range(max_new_tokens):
        if eos is not None and next_tok == eos:
            break
        generated.append(next_tok)

        cache_size = _get_cache_size(past_kv)
        if cache_size > budget:
            K, V = _extract_all_kv(past_kv, L)
            slots = evict_fn(K, V, tracker)

            # Correlation: importance rank percentile of chosen slot vs attention oracle
            if per_layer_imp is not None:
                T_imp = per_layer_imp.shape[1]
                for l, slot in enumerate(slots):
                    n = tracker.size(l)
                    imp_chosen = (
                        per_layer_imp[l, tracker.original_pos(l, slot)].item()
                        if tracker.original_pos(l, slot) < T_imp else 0.0
                    )
                    all_imps = [
                        per_layer_imp[l, tracker.original_pos(l, s)].item()
                        if tracker.original_pos(l, s) < T_imp else 0.0
                        for s in range(n)
                    ]
                    n_lower = sum(1 for imp in all_imps if imp < imp_chosen)
                    if n - 1 > 0:
                        correlation.append(n_lower / (n - 1))

            # Behavior: record the original position evicted and the absolute
            # sequence position at the time, per layer — used for recency /
            # sink-keeping diagnostics (orig_pos / true_pos ∈ [0,1) = recency).
            for l, slot in enumerate(slots):
                evicted.append((tracker.original_pos(l, slot), true_pos))

            _evict_per_layer(past_kv, slots)
            for l, s in enumerate(slots):
                tracker.evict(l, s)

        pos      = torch.tensor([[true_pos]], device=device)
        step_out = model(
            input_ids=torch.tensor([[next_tok]], device=device),
            past_key_values=past_kv,
            position_ids=pos,
            cache_position=pos.squeeze(0),
            use_cache=True,
        )
        next_tok = int(step_out.logits[0, -1].argmax())
        past_kv  = step_out.past_key_values
        tracker.append_all(true_pos)
        true_pos += 1

    return tokenizer.decode(generated, skip_special_tokens=True), correlation, evicted


# ── Misc helpers ──────────────────────────────────────────────────────────────

def score_full_cache(model, tokenizer, input_ids, gold, device, max_new_tokens):
    T = input_ids.shape[1]
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids.to(device),
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    text = tokenizer.decode(out[0, T:], skip_special_tokens=True)
    return flexible_extract(text, [gold])


def capture_per_layer_attention(model, input_ids, device, max_new_tokens=64):
    """Run reference generate with output_attentions=True → [L, T] per-layer importance.

    Returns None if output_attentions is unsupported (SDPA / flash_attention_2).
    """
    T          = input_ids.shape[1]
    n_layers   = model.config.num_hidden_layers
    n_kv_heads = getattr(model.config, "num_key_value_heads",
                         model.config.num_attention_heads)
    importance = torch.zeros(n_layers, T)
    counts     = torch.zeros(n_layers)

    try:
        with torch.no_grad():
            out = model.generate(
                input_ids=input_ids.to(device),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                output_attentions=True,
                return_dict_in_generate=True,
            )
        if not (hasattr(out, "attentions") and out.attentions):
            return None

        for step_attns in out.attentions:
            for l, layer_attn in enumerate(step_attns):
                if layer_attn is None or l >= n_layers:
                    continue
                seq_len  = layer_attn.shape[-1]
                n_prompt = min(T, seq_len)
                n_q      = layer_attn.shape[1]
                per_q    = layer_attn[0, :, 0, :n_prompt]
                if n_q % n_kv_heads == 0:
                    g = n_q // n_kv_heads
                    per_kv = per_q.view(n_kv_heads, g, n_prompt).amax(dim=1)
                    attn   = per_kv.mean(dim=0).cpu()
                else:
                    attn = per_q.mean(dim=0).cpu()
                importance[l, :n_prompt] += attn
                counts[l] += 1

    except (RuntimeError, ValueError, NotImplementedError):
        return None

    counts = counts.clamp(min=1).unsqueeze(1)
    importance /= counts
    importance /= importance.sum(dim=1, keepdim=True).clamp(min=1e-8)
    return importance  # [L, T]


def wilson_ci(successes: float, n: int, z: float = 1.96):
    if n == 0:
        return 0.0, 0.0
    p      = successes / n
    denom  = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return float(centre - margin), float(centre + margin)
