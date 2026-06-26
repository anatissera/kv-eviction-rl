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


def _finalize_episode(b, n_layers, ready,
                      ep_obs, ep_acts, ep_vals, ep_lps, ep_masks, ep_rews,
                      terminal_reward, bootstrap, last_v):
    """Stack per-episode accumulators, compute returns/advantages, append to ready."""
    T = len(ep_obs[b])
    obs_arr   = np.stack(ep_obs[b])    # [T, n_layers, *obs_shape]
    acts_arr  = np.stack(ep_acts[b])   # [T, n_layers]
    vals_arr  = np.stack(ep_vals[b])   # [T, n_layers]
    lps_arr   = np.stack(ep_lps[b])    # [T, n_layers]
    masks_arr = np.stack(ep_masks[b])  # [T, n_layers, max_len]
    rews_arr  = np.stack(ep_rews[b])   # [T, n_layers]

    if bootstrap:
        # Episode hit step cap — bootstrap from last value with gamma=1.
        rets_arr = np.zeros_like(vals_arr)
        nxt = last_v.copy()  # [n_layers]
        for t in reversed(range(T)):
            nxt = rews_arr[t] + nxt
            rets_arr[t] = nxt
    else:
        # Complete episode: return = terminal reward for every step (gamma=1).
        rets_arr = np.full_like(vals_arr, terminal_reward)

    advs_arr = rets_arr - vals_arr

    ready.append({
        'T':     T,
        'obs':   obs_arr,
        'acts':  acts_arr,
        'vals':  vals_arr,
        'lps':   lps_arr,
        'masks': masks_arr,
        'advs':  advs_arr,
        'rets':  rets_arr,
    })

    for lst in (ep_obs[b], ep_acts[b], ep_vals[b],
                ep_lps[b], ep_masks[b], ep_rews[b]):
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

    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps):
        assert self._last_obs is not None

        n_envs_total = env.num_envs       # N × n_layers
        N        = env.venv.n_parallel    # parallel episodes (e.g. 3)
        n_layers = n_envs_total // N      # transformer layers (28)

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

        obs        = self._last_obs
        wall_steps = 0

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
                    _finalize_episode(
                        b, n_layers, ready,
                        ep_obs, ep_acts, ep_vals, ep_lps, ep_masks, ep_rews,
                        terminal_reward=float(rewards[b * n_layers]),
                        bootstrap=False, last_v=None,
                    )

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

        new_buf = FP16ObsMaskableRolloutBuffer(
            buffer_size=total_steps,
            observation_space=self.observation_space,
            action_space=self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=n_layers,
        )
        new_buf.reset()

        row = 0
        for r in ready:
            T = r['T']
            sl = slice(row, row + T)
            new_buf.observations[sl] = r['obs']           # fp16 auto-cast on assign
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
