from sb3_contrib.ppo_recurrent import RecurrentPPO
import numpy as np

class RecurrentPPOParallel(RecurrentPPO):
    """
    RecurrentPPO subclass that supports continuous actions in parallel environments (n_envs > 1)
    by fixing the broadcasting issue of terminal values in the rollout buffer.

    This preserves all original SB3 tensor preprocessing.
    """

    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps):
        """
        Overrides collect_rollouts to reshape terminal_values before adding to rewards.
        """
        # Copy original method from RecurrentPPO
        # Only modify the line that adds bootstrapped terminal values to rewards
        rollout_buffer.reset()

        self._last_obs = self._last_obs if hasattr(self, "_last_obs") else env.reset()
        self._last_states = self._last_states if hasattr(self, "_last_states") else None
        self._episode_starts = self._episode_starts if hasattr(self, "_episode_starts") else np.ones(env.num_envs, dtype=bool)

        continue_training = True

        for _ in range(n_rollout_steps):
            actions, values, log_probs, states = self.policy.forward(
                self._last_obs, self._last_states, self._episode_starts
            )

            new_obs, rewards, dones, infos = env.step(actions)

            # Ensure rewards are 1D (n_envs,) in case env returns extra dims
            rewards = np.array(rewards).reshape(-1)

            # Compute terminal values for bootstrapping
            with np.errstate(all='ignore'):
                terminal_values = self.policy.predict_values(new_obs, self._last_states, self._episode_starts)
            # ---- FIX: squeeze extra dimensions ----
            terminal_values = np.squeeze(terminal_values, axis=-1)

            # Add step to rollout buffer
            rollout_buffer.add(
                self._last_obs,
                actions,
                rewards,
                self._last_states,
                terminal_values,
                log_probs,
                self._episode_starts
            )

            self._last_obs = new_obs
            self._last_states = states
            self._episode_starts = dones

            if callback is not None:
                continue_training = callback.on_step()
                if not continue_training:
                    break

        return continue_training