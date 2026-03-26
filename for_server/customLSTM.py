import torch
from torch import nn
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from sb3_contrib.ppo_recurrent import RecurrentPPO
from typing import Tuple, Union
from types import SimpleNamespace

# ----------------------------
# LayerNorm LSTM cell
# ----------------------------
class LayerNormLSTMCell(nn.Module):
    def __init__(
            self, 
            input_size: int, 
            hidden_size: int
        ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.W_x = nn.Linear(input_size, 4 * hidden_size, bias=False)
        self.W_h = nn.Linear(hidden_size, 4 * hidden_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(4 * hidden_size))

        self.ln_gates = nn.LayerNorm(4 * hidden_size)
        self.ln_c = nn.LayerNorm(hidden_size)

        print("NormLSTM updated successfully")

    def forward(
            self, 
            x: torch.Tensor, 
            state: Tuple[torch.Tensor, torch.Tensor]
        ):

        h, c = state
        gates = self.W_x(x) + self.W_h(h) + self.bias
        gates = self.ln_gates(gates)
        i, f, g, o = gates.chunk(4, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)

        c_new = f * c + i * g
        c_new = self.ln_c(c_new)
        h_new = o * torch.tanh(c_new)

        return h_new, c_new


class LayerNormLSTM(nn.Module):
    """
    1-layer LSTM with LayerNorm in the cell.

    Inputs:
      - x: (B, F) -> returns (B, H)
      - x: (T, B, F) -> returns (T, B, H)
    States (in/out): (h, c) with shape (1, B, H).
    """
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers  = 1
        self.cell = LayerNormLSTMCell(input_size, hidden_size)

    def forward(self, x: torch.Tensor, state: Tuple[torch.Tensor, torch.Tensor]):
        h, c = state  # (1,B,H)
        h, c = h[0], c[0]  # (B,H)

        if x.ndim == 2:  # (B,F) single step
            h, c = self.cell(x, (h, c))
            return h, (h.unsqueeze(0), c.unsqueeze(0))

        if x.ndim == 3:  # (T,B,F) sequence
            T, B, F = x.shape
            assert F == self.input_size, f"Expected input_size={self.input_size}, got {F}"
            outs = []
            for t in range(T):
                h, c = self.cell(x[t], (h, c))
                outs.append(h.unsqueeze(0))  # (1,B,H)
            y = torch.cat(outs, dim=0)  # (T,B,H)
            return y, (h.unsqueeze(0), c.unsqueeze(0))

        raise ValueError(f"Unexpected x.ndim {x.ndim}, expected 2 or 3")


# =========================
# Custom Recurrent Policy
# =========================

class CustomLayerNormLSTMPolicy(RecurrentActorCriticPolicy):
    """
    sb3_contrib-compatible recurrent policy that swaps in a LayerNorm LSTM.

    Supports:
      - shared_lstm (True/False)
      - enable_critic_lstm (for non-shared only)
      - n_lstm_layers = 1 (sb3_contrib limitation)

    Important design choices:
    - We replace the internal LSTM(s) inside __init__ *after* super().__init__ so
      the optimizer will include these parameters.
    - We do NOT rely on `lstm_input_dim` (which may not exist). We use `self.features_dim`.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Use the feature extractor output dimension as the LSTM input size.
        input_size = int(self.features_dim)  # robust source of feature size
        hidden_size = (
            getattr(self, "lstm_hidden_dim", None)  # new name (sb3 2.3+)
            or getattr(self, "lstm_hidden_size", None)  # old name
            or kwargs.get("lstm_hidden_size")  # last-resort fallback
        )
        if hidden_size is None:
            raise ValueError("Could not infer LSTM hidden size. Please provide lstm_hidden_size in policy_kwargs.")
        hidden_size = int(hidden_size)
        hidden_size = int(hidden_size)  # provided by policy_kwargs

        # Replace default LSTMs with our custom ones
        if self.shared_lstm:
            self.lstm = LayerNormLSTM(input_size, hidden_size)
        else:
            self.lstm_actor = LayerNormLSTM(input_size, hidden_size)
            if getattr(self, "enable_critic_lstm", True):
                self.lstm_critic = LayerNormLSTM(input_size, hidden_size)
            else:
                self.lstm_critic = None  # critic will bypass LSTM

    def _split_states(self, lstm_states):
             # Accept either sb3_contrib style (obj.pi/.vf) or tuple
        if hasattr(lstm_states, "pi"):
            return lstm_states.pi, lstm_states.vf
            # tuple: ((h_pi,c_pi), (h_vf,c_vf)) OR (h,c) for shared (we'll treat both)
        if isinstance(lstm_states, tuple) and len(lstm_states) == 2 and \
            isinstance(lstm_states[0], tuple) and isinstance(lstm_states[1], tuple):
            return lstm_states[0], lstm_states[1]
            # shared case passed as (h,c): use same for both
        return lstm_states, lstm_states

    def _pack_states(self, pi_state, vf_state):
            # sb3_contrib expects .pi and .vf
        return SimpleNamespace(pi=pi_state, vf=vf_state)
    
    def _ensure_state_batch(self, states, N: int, H: int, device, dtype):
        """
        Ensure lstm states have batch=N. If not, return fresh zeros.
        states: (h, c) each (1, B, H) or something else
        Return: (h, c) each (1, N, H)
        """
        if states is None:
            h = torch.zeros((1, N, H), device=device, dtype=dtype)
            c = torch.zeros((1, N, H), device=device, dtype=dtype)
            return (h, c)

        h, c = states
    # If already correct batch, keep
        if h.dim() == 3 and h.shape[0] == 1 and h.shape[2] == H and h.shape[1] == N:
            return (h, c)

    # Otherwise, make fresh zeros (B1: we do not carry states across minibatches)
        h_new = torch.zeros((1, N, H), device=device, dtype=h.dtype if h is not None else dtype)
        c_new = torch.zeros((1, N, H), device=device, dtype=c.dtype if c is not None else dtype)
        return (h_new, c_new)

    # ---- utilities ----
    @staticmethod
    def _reset_states_on_episode_start(states, episode_starts):
        """
        states: (h, c) each (1, B, H)
        episode_starts: (B,)
        Reset (zero) states where episode_starts == 1
        """
        h, c = states
        if episode_starts is None:
            return h, c
        mask_keep = (1.0 - episode_starts.float()).to(h.device)  # (B,)
        h0, c0 = h[0], c[0]  # (B,H)
        h0, c0 = h0 * mask_keep.view(-1, 1), c0 * mask_keep.view(-1, 1)
        return h0.unsqueeze(0), c0.unsqueeze(0)

    @staticmethod
    def _keep_mask_seq(episode_starts_tb):
        # episode_starts: (T,B) -> keep mask (T,B) with 1=keep, 0=reset
        return 1.0 - episode_starts_tb.float()

    def _step_one(self, lstm_module, x_bf, states, episode_starts_b):
        h, c = states  # (1,B,H)
        h, c = h[0], c[0]  # (B,H)

    # ---- Detect training mode: if sizes mismatch, skip masking ----
        if episode_starts_b is not None and episode_starts_b.shape[0] == h.shape[0]:
        # Rollout mode → apply reset mask
            mask_keep = (1.0 - episode_starts_b.float()).to(h.device)  # (B,)
            h = h * mask_keep.unsqueeze(1)
            c = c * mask_keep.unsqueeze(1)

    # ---- Forward LSTM ----
        if isinstance(lstm_module, LayerNormLSTM):
            y_bh, (hn, cn) = lstm_module(x_bf, (h.unsqueeze(0), c.unsqueeze(0)))
            return y_bh, (hn, cn)

    # Fallback for nn.LSTM
        if isinstance(lstm_module, nn.LSTM):
            y, (hn, cn) = lstm_module(
                x_bf.unsqueeze(1), (h.unsqueeze(0), c.unsqueeze(0))
            )
            return y[:, 0, :], (hn, cn)

        raise TypeError("Unknown LSTM module type")

    def _seq_many(self, lstm_module: Union[LayerNormLSTM, nn.LSTM], x_tbf: torch.Tensor, states, episode_starts_tb):
        """
        Sequence forward for training/evaluate_actions.
        x_tbf: (T, B, F), states: (1,B,H), episode_starts_tb: (T,B)
        """
        T, B, _ = x_tbf.shape
        h, c = states
        h, c = h[0], c[0]  # (B,H)
        keep_tb = self._keep_mask_seq(episode_starts_tb).to(x_tbf.device)  # (T,B)

        # Our custom module path
        if isinstance(lstm_module, LayerNormLSTM):
            outs = []
            for t in range(T):
                keep = keep_tb[t].view(-1, 1)  # (B,1)
                h = h * keep
                c = c * keep
                h, c = lstm_module.cell(x_tbf[t], (h, c))
                outs.append(h.unsqueeze(0))  # (1,B,H)
            y_tbh = torch.cat(outs, dim=0)  # (T,B,H)
            return y_tbh, (h.unsqueeze(0), c.unsqueeze(0))

        # Safety fallback: nn.LSTM path
        if isinstance(lstm_module, nn.LSTM):
            outs = []
            for t in range(T):
                keep = keep_tb[t].view(-1, 1)
                h = h * keep
                c = c * keep
                y1, (h1, c1) = lstm_module(x_tbf[t].unsqueeze(0), (h.unsqueeze(0), c.unsqueeze(0)))  # (1,B,H)
                h, c = h1[0], c1[0]
                outs.append(y1)  # (1,B,H)
            y_tbh = torch.cat(outs, dim=0)  # (T,B,H)
            return y_tbh, (h.unsqueeze(0), c.unsqueeze(0))

        raise TypeError("Unknown LSTM module type")

    # ----- sb3_contrib API: single-step inference used in collect_rollouts -----
    def forward(self, obs, lstm_states, episode_starts, deterministic: bool = False):
        device = self.device
        features = self.extract_features(obs).to(device)  # (B, F)

        if not hasattr(self, "_n_envs"):
            self._n_envs = int(episode_starts.shape[0])

        states_pi, states_vf = self._split_states(lstm_states)

        if self.shared_lstm:
            y_bh, new_state = self._step_one(self.lstm, features, states_pi, episode_starts)
            latent_pi = self.mlp_extractor.policy_net(y_bh)
            latent_vf = self.mlp_extractor.value_net(y_bh)
            out_states = self._pack_states(new_state, new_state)
        else:
        # actor
            y_pi_bh, new_states_pi = self._step_one(self.lstm_actor, features, states_pi, episode_starts)
        # critic
            if self.lstm_critic is not None:
                y_vf_bh, new_states_vf = self._step_one(self.lstm_critic, features, states_vf, episode_starts)
                vf_in = y_vf_bh
            else:
                new_states_vf = states_vf  # untouched
                vf_in = features

            latent_pi = self.mlp_extractor.policy_net(y_pi_bh)
            latent_vf = self.mlp_extractor.value_net(vf_in)
            out_states = self._pack_states(new_states_pi, new_states_vf)

        dist = self._get_action_dist_from_latent(latent_pi)
        actions = dist.get_actions(deterministic=deterministic)
        log_prob = dist.log_prob(actions)
        values = self.value_net(latent_vf)

        return actions, values, log_prob, out_states

    # ----- sb3_contrib API: sequence pass used in training -----
    def evaluate_actions(self, obs, actions, lstm_states, episode_starts):
        device = self.device

    # ---- Flatten observations to (N, obs_dim) regardless of input form ----
        if obs.ndim == 3:
            N = obs.shape[0] * obs.shape[1]
            obs_flat = obs.reshape(N, obs.shape[-1])
        elif obs.ndim == 2:
            N = obs.shape[0]
            obs_flat = obs
        else:
            raise RuntimeError(f"Unsupported obs.ndim={obs.ndim}. Expect 2 or 3.")

    # ---- Feature extraction on flat batch ----
        features = self.extract_features(obs_flat).to(device)  # (N, F)

    # ---- Split states from sampler (may NOT match N) ----
        states_pi, states_vf = self._split_states(lstm_states)

    # ---- B1: ignore episode starts inside training ----
        zeros_mask = torch.zeros(N, device=device, dtype=torch.float32)

    # ---- Ensure state batch matches N (otherwise create fresh zeros) ----
        if self.shared_lstm:
            H = int(self.lstm.hidden_size)
            states_pi = self._ensure_state_batch(states_pi, N, H, device, features.dtype)
        # shared: vf uses same states; we still compute latent_vf from the shared output
        else:
            H_pi = int(self.lstm_actor.hidden_size)
            states_pi = self._ensure_state_batch(states_pi, N, H_pi, device, features.dtype)
            if self.lstm_critic is not None:
                H_vf = int(self.lstm_critic.hidden_size)
                states_vf = self._ensure_state_batch(states_vf, N, H_vf, device, features.dtype)
            else:
            # critic bypasses LSTM; states_vf unused
                pass

    # ---- Run LSTM(s) in batch mode (single-step over the flat batch) ----
        if self.shared_lstm:
            y_bh, _ = self._step_one(self.lstm, features, states_pi, zeros_mask)  # (N, H)
            latent_pi = self.mlp_extractor.policy_net(y_bh)
            latent_vf = self.mlp_extractor.value_net(y_bh)
        else:
        # actor
            y_pi_bh, _ = self._step_one(self.lstm_actor, features, states_pi, zeros_mask)
            latent_pi = self.mlp_extractor.policy_net(y_pi_bh)
        # critic
            if self.lstm_critic is not None:
                y_vf_bh, _ = self._step_one(self.lstm_critic, features, states_vf, zeros_mask)
                latent_vf = self.mlp_extractor.value_net(y_vf_bh)
            else:
                latent_vf = self.mlp_extractor.value_net(features)

    # ---- Distribution, log_prob, entropy ----
        if actions.ndim >= 3:
            actions_flat = actions.reshape(N, *actions.shape[2:])
        elif actions.ndim == 2:
            actions_flat = actions.reshape(N, *actions.shape[1:])
        else:
            actions_flat = actions.view(N)

        dist = self._get_action_dist_from_latent(latent_pi)
        log_prob = dist.log_prob(actions_flat)   # (N,)
        entropy  = dist.entropy()                # (N,)
        values   = self.value_net(latent_vf)     # (N, 1)

    # B1: return exactly three tensors
        return values, log_prob, entropy


    # Ensure everything under policy.* is saved/loaded
    def _get_torch_save_params(self):
        return ["policy", "policy.optimizer"], []


# =========================
# (Optional) Thin wrapper
# =========================

class CustomRecurrentPPO(RecurrentPPO):
    """
    Wrapper around sb3_contrib.RecurrentPPO that defaults to CustomLayerNormLSTMPolicy
    if no policy is explicitly provided by the user.
    """
    def __init__(self, *args, policy=None, **kwargs):
        if policy is None:
            policy = CustomLayerNormLSTMPolicy
        # Call parent with policy as the first positional argument
        super().__init__(policy, *args, **kwargs)
