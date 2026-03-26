import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical

class DummyReLUFeatureExtractor(nn.Module):
    def __init__(
            self, 
            input_dim, 
            features_dim0: int = 64, 
            features_dim1: int = 128,
            features_dim2: int = 256,
            features_dim3: int = 256
        ):
        super().__init__()
        self.pre_layer = nn.Sequential(
            nn.Flatten(),                 
            nn.Linear(input_dim, features_dim0),
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
            obs_dim: int = 32, 
            action_dim: int = 4
        ):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.features_extractor = DummyReLUFeatureExtractor(obs_dim)

        self.lstm_actor = nn.LSTM(
            input_size=256,
            hidden_size=256,
            num_layers=1,
            batch_first=True
        )

        self.policy_net = nn.Sequential(
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU()
        )

        self.action_net = nn.Linear(128, action_dim)

    def init_hidden(self, batch_size=1, device="cpu"):
        h = torch.zeros(1, batch_size, 256).to(device)
        c = torch.zeros(1, batch_size, 256).to(device)

        #self.to(device)
        
        return (h, c)
    
    def forward(self, obs, hidden):
        if not torch.is_tensor(obs):
            obs = torch.as_tensor(obs, dtype=torch.float32, device=hidden[0].device)
        else:
            obs = obs.to(dtype=torch.float32, device=hidden[0].device)

        if obs.ndim == 2:   
            obs = obs.unsqueeze(1)  
        elif obs.ndim == 1: 
            obs = obs.unsqueeze(0).unsqueeze(1) 

        batch, seq_len, last = obs.shape

        if last != self.obs_dim:
            raise ValueError(f"Expected obs last-dim {self.obs_dim}, got {last}")

        feats = self.features_extractor(obs.view(-1, self.obs_dim))
        feats = feats.view(batch, seq_len, -1)

        lstm_out, new_hidden = self.lstm_actor(feats, hidden)

        x = self.policy_net(lstm_out)
        logits = self.action_net(x)  
        return logits, new_hidden
    
    def act(self, obs, hidden, epsilon = None, confidence_level = None, device = "cpu"):
        if epsilon is not None and confidence_level is not None:
            return self.act_without_exploration(obs=obs, hidden=hidden, epsilon=epsilon, confidence_level=confidence_level, device=device)
        else:
            return self.act_without_exploration(obs=obs, hidden=hidden, device=device)

    def act_without_exploration(self, obs, hidden, device = "cpu"):
        if isinstance(obs, np.ndarray):
            obs = torch.as_tensor(obs, dtype=torch.float32, device=device)

        logits, new_hidden = self.forward(obs, hidden)
        logits = logits[:, -1, :]

        dist = Categorical(logits=logits)
        action = torch.argmax(logits, dim=1)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, log_prob, entropy, new_hidden
    
    def act_with_exploration(self, obs, hidden, epsilon = 0.05, confidence_level = 0.99, device = "cpu"):
        if isinstance(obs, np.ndarray):
            obs = torch.as_tensor(obs, dtype=torch.float32, device=device)

        logits, new_hidden = self.forward(obs, hidden)
        logits = logits[:, -1, :]

        dist = Categorical(logits=logits)

        probs = torch.softmax(logits, dim=-1)
        greedy_action = torch.argmax(probs, dim=-1)

        if greedy_action.item() == 0 and probs[0, 0] > confidence_level:
            action = greedy_action
        else:
            if np.random.rand() < epsilon:
                #print(1)
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