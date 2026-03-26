import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces

class ReLUFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim0: int = 64, features_dim1: int = 256):
        super().__init__(observation_space, features_dim1)
        if not isinstance(observation_space, spaces.Box):
            raise ValueError("ReLUFeatureExtractor only supports Box observation spaces")

        n_flatten = int(np.prod(observation_space.shape))
        self.pre_layer = nn.Sequential(
            nn.Flatten(),                 
            nn.Linear(n_flatten, features_dim0),
            nn.LayerNorm(features_dim0),
            nn.GELU(),
            nn.Linear(features_dim0, features_dim1),
            nn.LayerNorm(features_dim1),
            nn.GELU(),
            nn.LayerNorm(features_dim1)
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        obs = observations
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        obs = obs.to(dtype=torch.float32)
        return self.pre_layer(obs)