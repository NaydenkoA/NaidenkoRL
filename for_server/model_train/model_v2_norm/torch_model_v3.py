import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical
from typing import Tuple

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
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
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

class DummyReLUFeatureExtractor(nn.Module):
    def __init__(
            self, 
            observation_space = 16, 
            features_dim0: int = 32, 
            features_dim1: int = 64,
            features_dim2: int = 128,
            features_dim3: int = 256
        ):
        super().__init__()
        self.pre_layer = nn.Sequential(
            nn.Flatten(),                 
            nn.Linear(observation_space, features_dim0),
            nn.LayerNorm(features_dim0),
            nn.GELU(),
            nn.Linear(features_dim0, features_dim1),
            nn.LayerNorm(features_dim1),
            nn.GELU(),
            nn.Linear(features_dim1, features_dim2),
            nn.LayerNorm(features_dim2),
            nn.GELU(),
            nn.Linear(features_dim2, features_dim3),
            nn.LayerNorm(features_dim3),
            nn.GELU(),
            nn.LayerNorm(features_dim3)
        )


    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(obs):
            obs = torch.as_tensor(obs)
        obs = obs.to(dtype=torch.float32)
        return self.pre_layer(obs)

class reccurentActor(nn.Module):
    def __init__(
            self, 
            obs_dim: int = 16, 
            action_dim: int = 4
        ):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.features_extractor = DummyReLUFeatureExtractor(observation_space = obs_dim)

        self.lstm_actor = LayerNormLSTM(
            input_size=256,
            hidden_size=256
        )

        self.policy_net = nn.Sequential(
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU()
        )

        self.action_net = nn.Linear(128, action_dim)

    def init_hidden(self, batch_size=1, device="cpu"):
        h = torch.zeros(1, batch_size, 256, device=device)
        c = torch.zeros(1, batch_size, 256, device=device)
        return (h, c)

    def forward(self, obs, hidden):
        obs = torch.as_tensor(obs, dtype=torch.float32)

        if obs.ndim != 3:
            raise ValueError(f"Expected obs shape (B,T,{self.obs_dim}), got {obs.shape}")

        B, T, _ = obs.shape

        feats = self.features_extractor(obs.view(B*T, self.obs_dim))
        feats = feats.view(B, T, -1)               

        feats = feats.transpose(0, 1)

        lstm_out, new_hidden = self.lstm_actor(feats, hidden)  

        lstm_out = lstm_out.transpose(0, 1)      

        x = self.policy_net(lstm_out)
        logits = self.action_net(x)
        return logits, new_hidden
    
    def act(self, obs, hidden, epsilon = None, confidence_level = None, device = "cpu"):
        if epsilon is not None and confidence_level is not None:
            return self.act_with_exploration(obs=obs, hidden=hidden, epsilon=epsilon, confidence_level=confidence_level, device=device)
        else:
            return self.act_without_exploration(obs=obs, hidden=hidden, device=device)

    def act_without_exploration(self, obs, hidden, device="cpu"):
        if isinstance(obs, np.ndarray):
            obs = torch.as_tensor(obs, dtype=torch.float32, device=device)

        if obs.ndim == 1:
            obs = obs.unsqueeze(0).unsqueeze(0)

        h, c = hidden
        hidden = (h.to(device), c.to(device))

        
        logits, new_hidden = self.forward(obs, hidden)

        logits = logits[:, -1, :]

        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, log_prob, entropy, new_hidden
    
    def act_with_exploration(self, obs, hidden, epsilon=0.05, confidence_level=0.99, device="cpu"):
        if isinstance(obs, np.ndarray):
            obs = torch.as_tensor(obs, dtype=torch.float32, device=device)

        if obs.ndim == 1:
            obs = obs.unsqueeze(0).unsqueeze(0)

        h, c = hidden
        hidden = (h.to(device), c.to(device))

        with torch.no_grad():
            logits, new_hidden = self.forward(obs, hidden)

        logits = logits[:, -1, :]

        dist = Categorical(logits=logits)
        probs = torch.softmax(logits, dim=-1)
        greedy_action = torch.argmax(probs, dim=-1)

        if greedy_action.item() == 0 and probs[0, 0] > confidence_level:
            action = greedy_action
        else:
            if np.random.rand() < epsilon:
                action = torch.randint(0, self.action_dim, (1,), device=device)
            else:
                action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, log_prob, entropy, new_hidden
    
    def set_initial_state(self, file_path, device = "cpu"):
        state = torch.load(file_path, map_location = device, weights_only=True)

        self.features_extractor.load_state_dict(state["features_extractor"])
        self.lstm_actor.load_state_dict(state["lstm_actor"])
        self.policy_net.load_state_dict(state["policy_net"])
        self.action_net.load_state_dict(state["action_net"])

        self.to(device)

        self.eval()