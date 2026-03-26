import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical
import torch.nn.functional as F

class actor_v1(nn.Module):
    def __init__(
        self, 
        obs_dim: int = 32,
        action_dim: int = 4,
        device = "cpu"
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device

        self.extractor = nn.Sequential(
            nn.Linear(self.obs_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.GELU()
        )

        self.lstm = nn.LSTM(
            input_size = 256,
            hidden_size = 256,
            num_layers = 1,
            batch_first = True
        )

        self.policy_net = nn.Sequential(
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU()
        )

        self.action_net = nn.Linear(256, self.action_dim)

        self.to(self.device)

    def init_hidden(self, batch_size = 1):
        h = torch.zeros(1, batch_size, 256).to(self.device)
        c = torch.zeros(1, batch_size, 256).to(self.device)
        
        return (h, c)
    
    def forward_single(self, obs, hidden, with_alt_entropy = False, forced_action = None):
        if not torch.is_tensor(obs):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)

        assert obs.ndim == 1, "Input must be 1-dimensional!"

        x = self.extractor(obs)

        x = x.unsqueeze(0).unsqueeze(1) 
        x, hidden = self.lstm(x, hidden)
        x = x.squeeze() #removes all dims with size = 1

        x = self.policy_net(x)

        raw_params = self.action_net(x)
        dist = Categorical(logits = raw_params)

        if forced_action is None:
            action = dist.sample()
        else:
            if not torch.is_tensor(forced_action):
                action = torch.as_tensor(forced_action, dtype=torch.long, device=self.device)
            else:
                action = forced_action

            assert action.ndim == 0, "Action must be a scalar!"

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        if with_alt_entropy and (self.action_dim > 2): 
            subset = raw_params[1 : self.action_dim]
            subset_dist = Categorical(logits = subset)
            entropy_alt = subset_dist.entropy()
        else:
            entropy_alt = None

        return action, hidden, log_prob, entropy, entropy_alt
    
    def forward_block(self, obs, hidden, with_alt_entropy = False, forced_action = None):
        if not torch.is_tensor(obs):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)

        assert obs.ndim == 2, "Input must be 2-dimensional!"

        x = self.extractor(obs)

        x = x.unsqueeze(0)
        x, hidden = self.lstm(x, hidden)
        x = x.squeeze(0)

        x = self.policy_net(x)

        raw_params = self.action_net(x)
        dist = Categorical(logits = raw_params)
        
        if forced_action is None:
            action = dist.sample()
        else:
            if not torch.is_tensor(forced_action):
                action = torch.as_tensor(forced_action, dtype=torch.long, device=self.device)
            else:
                action = forced_action

            assert action.shape == (obs.shape[0],), "Action shape must be (T,)!"

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        if with_alt_entropy and (self.action_dim > 2): 
            subset = raw_params[:, 1 : self.action_dim]
            subset_dist = Categorical(logits = subset)
            entropy_alt = subset_dist.entropy()
        else:
            entropy_alt = None

        return action, hidden, log_prob, entropy, entropy_alt

    def forward_batch_singles(self, obs, hidden, with_alt_entropy = False, forced_action = None):
        if not torch.is_tensor(obs):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)

        assert obs.ndim == 2, "Input must be 2-dimensional!"

        x = self.extractor(obs)

        x = x.unsqueeze(1)
        x, hidden = self.lstm(x, hidden)
        x = x.squeeze(1)

        x = self.policy_net(x)

        raw_params = self.action_net(x)
        dist = Categorical(logits = raw_params)
        
        if forced_action is None:
            action = dist.sample()
        else:
            if not torch.is_tensor(forced_action):
                action = torch.as_tensor(forced_action, dtype=torch.long, device=self.device)
            else:
                action = forced_action

            assert action.shape == (obs.shape[0],), "Action shape must be (B,)!"

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        if with_alt_entropy and (self.action_dim > 2): 
            subset = raw_params[:, 1 : self.action_dim]
            subset_dist = Categorical(logits = subset)
            entropy_alt = subset_dist.entropy()
        else:
            entropy_alt = None

        return action, hidden, log_prob, entropy, entropy_alt
    
    def forward_batch(self, obs, hidden, with_alt_entropy = False, forced_action = None):
        if not torch.is_tensor(obs):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)

        assert obs.ndim == 3, "Input must be 3-dimensional!"

        B, T, _ = obs.shape

        x = obs.flatten(0, 1)   # merge dims 0 and 1
        x = self.extractor(x)
        x = x.reshape(B, T, -1)

        x, hidden = self.lstm(x, hidden)

        x = x.flatten(0, 1)
        x = self.policy_net(x)

        raw_params = self.action_net(x).reshape(B, T, self.action_dim)
        dist = Categorical(logits = raw_params)
        
        if forced_action is None:
            action = dist.sample()
        else:
            if not torch.is_tensor(forced_action):
                action = torch.as_tensor(forced_action, dtype=torch.long, device=self.device)
            else:
                action = forced_action

            assert action.shape == (obs.shape[0], obs.shape[1]), "Action shape must be (B, T)!"

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        if with_alt_entropy and (self.action_dim > 2): 
            subset = raw_params[:, :, 1 : self.action_dim]
            subset_dist = Categorical(logits = subset)
            entropy_alt = subset_dist.entropy()
        else:
            entropy_alt = None

        return action, hidden, log_prob, entropy, entropy_alt
    
    def forward_single_with_exploration(self, obs, hidden, with_alt_entropy = False, epsilon = None, confidence_level = 0.99):
        if (epsilon is None) or (epsilon == 0):
            return self.forward_single(obs, hidden, with_alt_entropy)
        
        if not torch.is_tensor(obs):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)

        assert obs.ndim == 1, "Input must be 1-dimensional!"

        x = self.extractor(obs)

        x = x.unsqueeze(0).unsqueeze(1) 
        x, hidden = self.lstm(x, hidden)
        x = x.squeeze() 

        x = self.policy_net(x)

        raw_params = self.action_net(x)
        dist = Categorical(logits = raw_params)
        prim_action = torch.argmax(raw_params)

        if (prim_action.item() != 0) or (dist.probs[0].item() < confidence_level):
            if torch.rand(1).item() < epsilon:
                action = torch.randint(0, self.action_dim, (), device = self.device)
            else:
                action = prim_action
        else:
            action = prim_action

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        if with_alt_entropy and (self.action_dim > 2): 
            subset = raw_params[1 : self.action_dim]
            subset_dist = Categorical(logits = subset)
            entropy_alt = subset_dist.entropy()
        else:
            entropy_alt = None

        return action, hidden, log_prob, entropy, entropy_alt
        
    def init_forget_gate_bias(self, value=0.8, layer=0):
        H = self.lstm.hidden_size
        fs = H
        fe = 2 * H

        with torch.no_grad():
            getattr(self.lstm, f"bias_ih_l{layer}")[fs:fe].fill_(value)
            getattr(self.lstm, f"bias_hh_l{layer}")[fs:fe].fill_(value)

    def act(self, obs, hidden, deterministic = True):
        with torch.no_grad():
            if not torch.is_tensor(obs):
                obs = torch.tensor(obs, dtype=torch.float32, device=self.device)

            assert obs.ndim == 1, "Input must be 1-dimensional!"

            x = self.extractor(obs)

            x = x.unsqueeze(0).unsqueeze(1) 
            x, hidden = self.lstm(x, hidden)
            x = x.squeeze() 

            x = self.policy_net(x)

            raw_params = self.action_net(x)
            dist = Categorical(logits = raw_params)

            if deterministic:
                action = torch.argmax(raw_params)
            else:
                action = dist.sample()

            log_probs = dist.log_prob(action)
            entropy = dist.entropy()
        
        return action.item(), hidden, log_probs, entropy


class critic_v1(nn.Module):
    def __init__(
        self,
        obs_dim: int = 32,
        device = "cpu"
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.device = device

        self.extractor = nn.Sequential(
            nn.Linear(self.obs_dim, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Linear(32, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.GELU()
        )

        self.lstm = nn.LSTM(
            input_size = 256,
            hidden_size = 256,
            num_layers = 1,
            batch_first = True
        )

        self.policy_net = nn.Sequential(
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
        )

        self.value_net = nn.Linear(256, 1) 

        self.to(self.device)

    def forward_single(self, obs, hidden):
        if not torch.is_tensor(obs):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)

        x = self.extractor(obs)

        x = x.unsqueeze(0).unsqueeze(1) 
        x, hidden = self.lstm(x, hidden)
        x = x.squeeze() 

        x = self.policy_net(x)
        value = self.value_net(x)

        return value, hidden
    
    def forward_block(self, obs, hidden):
        if not torch.is_tensor(obs):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)

        x = self.extractor(obs)
            
        x = x.unsqueeze(0)
        x, hidden = self.lstm(x, hidden)
        x = x.squeeze(0)

        x = self.policy_net(x)
        value = self.value_net(x)

        return value, hidden
    
    def forward_batch_singles(self, obs, hidden):
        if not torch.is_tensor(obs):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)

        x = self.extractor(obs)
            
        x = x.unsqueeze(1)
        x, hidden = self.lstm(x, hidden)
        x = x.squeeze(1)

        x = self.policy_net(x)
        value = self.value_net(x)

        return value, hidden
    
    def forward_batch(self, obs, hidden):
        if not torch.is_tensor(obs):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)

        B, T, _ = obs.shape
        
        x = obs.flatten(0, 1)   # merge dims 0 and 1
        x = self.extractor(x)
        x = x.reshape(B, T, -1)

        x, hidden = self.lstm(x, hidden)
        x = obs.flatten(0, 1)

        x = self.policy_net(x)

        value = self.value_net(x).reshape(B, T, -1)

        return value, hidden
    
    def init_forget_gate_bias(self, value=0.8, layer=0):
        H = self.lstm.hidden_size
        fs = H
        fe = 2 * H

        with torch.no_grad():
            getattr(self.lstm, f"bias_ih_l{layer}")[fs:fe].fill_(value)
            getattr(self.lstm, f"bias_hh_l{layer}")[fs:fe].fill_(value)

    def init_hidden(self, batch_size = 1):
        h = torch.zeros(1, batch_size, 256).to(self.device)
        c = torch.zeros(1, batch_size, 256).to(self.device)
        
        return (h, c)

