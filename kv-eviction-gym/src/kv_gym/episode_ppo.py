"""EpisodeMaskablePPO: MaskablePPO that collects complete episodes per rollout.

Each of the N parallel episodes contributes exactly one complete episode to the
buffer. Once episode b finishes, no further transitions from b are added for
this rollout — preventing the shorter episodes' "second episode" transitions
from polluting the buffer.

With gamma=1, gae_lambda=1: advantage[t] = R_terminal - V(s_t) for all t in a
complete episode. This is computed immediately on episode completion. Episodes
that hit the step cap fall back to bootstrapped returns.

Buffer is dynamically sized to (sum of episode lengths) x n_layers per rollout,
laid out as N episodes concatenated along the time axis.
"""

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.utils import obs_as_tensor
from kv_gym.buffer import FP16ObsMaskableRolloutBuffer

def _rss_mb():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) // 1024
    except Exception:
        pass
    return -1


def _finalize_episode(b, n_layers, ready,
                      ep_obs, ep_acts, ep_vals, ep_lps, ep_masks, ep_rews,
                      terminal_reward, bootstrap, last_v):
    """Move per-episode obs list to ready, compute returns/advantages, append to ready.

    Obs are NOT stacked here — the list of T arrays is moved directly to ready
    and written item-by-item in the buffer assembly loop (with per-item freeing).
    This eliminates the np.stack 2× peak: stacking would require the full source
    list (up to 11 GB) and the full dest array (11 GB) simultaneously in RAM.
    With the list-move approach the only peak is N × one_episode_obs (already
    resident) with no transient copy.
    """
    T = len(ep_obs[b])

    # Move obs list instead of stacking — zero extra allocation.
    obs_list  = ep_obs[b]
    ep_obs[b] = []          # replace with empty; old list now owned by ready

    acts_arr  = np.stack(ep_acts[b])   # [T, n_layers] — negligible size
    vals_arr  = np.stack(ep_vals[b])   # [T, n_layers]
    lps_arr   = np.stack(ep_lps[b])    # [T, n_layers]
    masks_arr = np.stack(ep_masks[b])  # [T, n_layers, max_len] — ~44 MB at T=650
    rews_arr  = np.stack(ep_rews[b])   # [T, n_layers]

    if bootstrap:
        # Episode hit step cap — bootstrap from last value with gamma=1.
        rets_arr = np.zeros_like(vals_arr)
        nxt = last_v.copy()  # [n_layers]
        for t in reversed(range(T)):
            nxt = rews_arr[t] + nxt
            rets_arr[t] = nxt
    else:
        # Complete episode, gamma=1: G_t = Σ_{k=t}^{T} r_k (suffix sum of rewards).
        # With only a sparse terminal reward, rews_arr[t]=0 for t<T and the suffix
        # sum collapses to the constant terminal_reward — same as before.
        # With dense per-step rewards (e.g. entropy_reward_weight > 0), this
        # correctly propagates each step's reward backward to all earlier timesteps.
        rets_arr = np.zeros_like(vals_arr)
        nxt = np.zeros(rews_arr.shape[1], dtype=rews_arr.dtype)  # [n_layers]
        for t in reversed(range(T)):
            nxt = rews_arr[t] + nxt
            rets_arr[t] = nxt

    advs_arr = rets_arr - vals_arr

    ready.append({
        'T':        T,
        'obs_list': obs_list,   # list of T arrays, each (n_layers, *obs_shape)
        'acts':     acts_arr,
        'vals':     vals_arr,
        'lps':      lps_arr,
        'masks':    masks_arr,
        'advs':     advs_arr,
        'rets':     rets_arr,
    })

    # obs already moved; clear the rest
    for lst in (ep_acts[b], ep_vals[b], ep_lps[b], ep_masks[b], ep_rews[b]):
        lst.clear()


class EpisodeMaskablePPO(MaskablePPO):
    """MaskablePPO variant that waits for complete episodes before updating.

    n_steps (from config) is used as a safety cap — if an episode hasn't
    finished by then, it's bootstrapped. In normal operation all episodes
    finish before the cap.
    """

    def _setup_model(self) -> None:
        # MaskablePPO.__init__ allocates a default rollout buffer of shape
        # (n_steps, n_envs, *obs_shape). With n_steps=650 and n_envs=140 that
        # is ~108 GB. Since collect_rollouts replaces self.rollout_buffer each
        # rollout with a correctly-sized dynamic buffer, we only need a 1-row
        # placeholder here.
        real_n_steps = self.n_steps
        self.n_steps = 1
        super()._setup_model()
        self.n_steps = real_n_steps

        # Repeat-problem support: train on the same N problems for this many
        # consecutive rollouts before sampling new ones. Set from train.py.
        self.repeats_per_problem: int = 1
        self._problem_repeat_idx: int = 0

    def train(self) -> None:
        # Free the rollout buffer immediately after training rather than at the
        # start of the next collect_rollouts. The next rollout builds a fresh
        # buffer anyway, so holding this one (up to 14 GB) through the entire
        # next collection doubles peak RSS unnecessarily.
        super().train()
        self.rollout_buffer = None
        print(f"  [mem] after train, buf freed: {_rss_mb()} MB", flush=True)

    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps, use_masking=True):
        assert self._last_obs is not None

        # Free the previous rollout buffer BEFORE accumulating ep_obs lists.
        # Must clear BOTH the instance attribute AND the local parameter: Python's
        # refcounting only frees an object when ALL references drop. The caller
        # passes self.rollout_buffer as the `rollout_buffer` argument, so setting
        # only self.rollout_buffer = None leaves the local param holding the old
        # buffer (12+ GB) alive for the entire collection — causing OOM when
        # ep_obs lists fill up.
        self.rollout_buffer = None
        rollout_buffer = None  # drop local ref so old buffer is freed immediately
        print(f"  [mem] after freeing old buf: {_rss_mb()} MB", flush=True)

        n_envs_total = env.num_envs       # N × n_layers
        N        = env.venv.n_parallel    # parallel episodes (e.g. 3)
        n_layers = n_envs_total // N      # transformer layers (28)

        # ── Re-align all N slots at the start of every rollout ─────────────
        # CRITICAL for bounded memory. env.reset() samples a fresh SHARED budget
        # and re-prefills all N slots so they start at identical cache_size and
        # finish close together (T0 ≈ T1). Without this, slots desynchronize
        # after iteration 1: each slot is reset INDEPENDENTLY inside step_wait()
        # at a different wall-step and with the stale shared budget, so in
        # iteration 2+ one slot finishes far earlier than the other. The PPO
        # loop runs until BOTH finish (all(ep_done)), so the surviving slot
        # pushes total_steps = T0 + T1 toward 2 × n_steps_cap while the env keeps
        # materializing fp32 observations for the restarted "phantom" episodes
        # every wall-step — the source of the iteration-2 RSS blow-up. Resetting
        # here makes every rollout behave like iteration 1 (buffer_rows ≈ 678).
        #
        # Repeat-problem: for the first rollout of each problem-group, reset to
        # new problems (_repeat_episode=False). For subsequent rollouts in the
        # same group, reset to the same problems (_repeat_episode=True) so the
        # policy gets multiple gradient updates on the same task.
        is_repeat = (self._problem_repeat_idx > 0)
        env.venv._repeat_episode = is_repeat
        self._problem_repeat_idx = (self._problem_repeat_idx + 1) % self.repeats_per_problem
        print(f"  [repeat] {'same' if is_repeat else 'new'} problems "
              f"(idx={self._problem_repeat_idx - 1 if is_repeat else 0}/"
              f"{self.repeats_per_problem})", flush=True)
        obs = env.reset()
        self._last_obs = obs

        self.policy.set_training_mode(False)
        callback.on_rollout_start()

        # ── Per-episode accumulators ──────────────────────────────────────
        ep_obs   = [[] for _ in range(N)]
        ep_acts  = [[] for _ in range(N)]
        ep_vals  = [[] for _ in range(N)]
        ep_lps   = [[] for _ in range(N)]
        ep_masks = [[] for _ in range(N)]
        ep_rews  = [[] for _ in range(N)]
        ep_done  = [False] * N

        ready = []  # completed episode dicts (advantages already computed)

        # obs already set from env.reset() above
        wall_steps = 0

        print(f"  [mem] before loop start: {_rss_mb()} MB", flush=True)
        while not all(ep_done) and wall_steps < n_rollout_steps:
            action_masks = get_action_masks(env)  # [N*n_layers, max_len]

            with torch.no_grad():
                obs_t = obs_as_tensor(obs, self.device)
                actions, values, log_probs = self.policy.forward(
                    obs_t, action_masks=action_masks
                )
            acts_np = actions.cpu().numpy()          # [N*n_layers]
            vals_np = values.cpu().numpy().flatten() # [N*n_layers]
            lps_np  = log_probs.cpu().numpy()        # [N*n_layers]

            new_obs, rewards, dones, infos = env.step(acts_np)
            wall_steps += 1

            # Count only transitions from still-running episodes
            useful = sum(n_layers for b in range(N) if not ep_done[b])
            self.num_timesteps += useful

            # Populate ep_info_buffer so TrainingLogger / SB3 rollout stats work.
            # Standard collect_rollouts calls this; our override must too.
            self._update_info_buffer(infos, dones)

            callback.update_locals(locals())
            if not callback.on_step():
                return False

            for b in range(N):
                if ep_done[b]:
                    continue
                sl = slice(b * n_layers, (b + 1) * n_layers)
                ep_obs[b].append(obs[sl].astype(np.float16))
                ep_acts[b].append(acts_np[sl])
                ep_vals[b].append(vals_np[sl])
                ep_lps[b].append(lps_np[sl])
                ep_masks[b].append(action_masks[sl])
                ep_rews[b].append(rewards[sl])

                # All n_layers envs share the same done flag for episode b
                if dones[b * n_layers]:
                    ep_done[b] = True
                    ep_len = len(ep_obs[b])  # len before _finalize moves the list
                    _finalize_episode(
                        b, n_layers, ready,
                        ep_obs, ep_acts, ep_vals, ep_lps, ep_masks, ep_rews,
                        terminal_reward=float(rewards[b * n_layers]),
                        bootstrap=False, last_v=None,
                    )
                    print(f"  [mem] ep{b} done T={ep_len} rss={_rss_mb()} MB", flush=True)

            if wall_steps % 50 == 0:
                lens = [len(ep_obs[b]) for b in range(N)]
                print(f"  [mem] step={wall_steps} ep_lens={lens} rss={_rss_mb()} MB", flush=True)

            obs = new_obs

        self._last_obs = obs
        self._last_episode_starts = np.zeros(n_envs_total, dtype=bool)

        # ── Bootstrap incomplete episodes (hit step cap) ──────────────────
        if not all(ep_done):
            with torch.no_grad():
                obs_t = obs_as_tensor(obs, self.device)
                _, last_vals, _ = self.policy.forward(obs_t)
            lv_np = last_vals.cpu().numpy().flatten()
            for b in range(N):
                if ep_done[b] or not ep_obs[b]:
                    continue
                sl = slice(b * n_layers, (b + 1) * n_layers)
                _finalize_episode(
                    b, n_layers, ready,
                    ep_obs, ep_acts, ep_vals, ep_lps, ep_masks, ep_rews,
                    terminal_reward=None, bootstrap=True, last_v=lv_np[sl],
                )

        # ── Assemble buffer: N episodes concatenated along time axis ──────
        # Layout: episode 0 (T0 rows) | episode 1 (T1 rows) | episode 2 (T2 rows)
        # Shape: [T0+T1+T2, n_layers]. No padding, no wasted rows.
        total_steps = sum(r['T'] for r in ready)
        ep_lens = [r['T'] for r in ready]
        print(f"  [mem] before buf alloc: rss={_rss_mb()} MB  ep_lens={ep_lens}  total_steps={total_steps}", flush=True)

        new_buf = FP16ObsMaskableRolloutBuffer(
            buffer_size=total_steps,
            observation_space=self.observation_space,
            action_space=self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=n_layers,
        )
        print(f"  [mem] after buf __init__: rss={_rss_mb()} MB", flush=True)
        # NOTE: do NOT call new_buf.reset() here. FP16ObsMaskableRolloutBuffer
        # __init__ already calls reset() via the BaseBuffer parent chain, which
        # allocates the (total_steps, n_layers, *obs_shape) fp16 observations
        # array. Calling reset() again allocates a SECOND such array and orphans
        # the first. Both are np.empty (demand-paged, ~0 RSS until written), so
        # the double allocation does not by itself cause OOM, but it is wasteful
        # virtual address space and an easy footgun — one explicit allocation is
        # enough.
        print(f"  [mem] after buf reset: rss={_rss_mb()} MB", flush=True)

        row = 0
        for r in ready:
            T = r['T']
            sl = slice(row, row + T)
            # Copy obs item-by-item, freeing each source item immediately via
            # refcount. new_buf.observations is np.empty (demand-paged), so only
            # the pages being written right now are physical. Combined with the
            # per-item free, the peak for (all obs_lists) + (buffer obs written
            # so far) stays bounded at N × one_episode_obs, not 2× per episode.
            obs_list = r.pop('obs_list')
            for i in range(T):
                new_buf.observations[row + i] = obs_list[i]
                obs_list[i] = None    # refcount → 0 → freed immediately
            del obs_list
            print(f"  [mem] after copy ep T={T}: rss={_rss_mb()} MB", flush=True)
            new_buf.actions[sl]      = r['acts'][..., None]  # [T, n_layers, 1]
            new_buf.values[sl]       = r['vals']
            new_buf.log_probs[sl]    = r['lps']
            new_buf.action_masks[sl] = r['masks']
            new_buf.advantages[sl]   = r['advs']
            new_buf.returns[sl]      = r['rets']
            # episode_starts: mark first step of each episode
            new_buf.episode_starts[row] = True
            row += T

        new_buf.full = True
        new_buf.pos  = total_steps
        self.rollout_buffer = new_buf

        n_complete = sum(ep_done)
        n_cap      = N - n_complete
        print(f"  [episode rollout] wall_steps={wall_steps}  "
              f"complete={n_complete}/{N}  bootstrapped={n_cap}  "
              f"buffer_rows={total_steps}")

        callback.on_rollout_end()
        return True
