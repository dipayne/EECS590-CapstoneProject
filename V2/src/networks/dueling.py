"""Dueling Q-Network architecture (Wang et al., 2016).

The network decomposes Q(s,a) into:
    Q(s,a) = V(s) + A(s,a) − mean_a(A(s,a))

This decomposition allows the network to learn state-value V(s) and
action-advantage A(s,a) separately, which greatly stabilises learning
in environments where many actions have similar values and the choice
of action only matters in a few critical states.

Reference: Wang et al., "Dueling Network Architectures for Deep
Reinforcement Learning," ICML 2016.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from typing import Tuple

from src.networks.mlp import _orthogonal_init


class DuelingQNetwork(nn.Module):
    """Dueling deep Q-network with shared feature extractor.

    Architecture:
        obs → shared_backbone → [value_stream | advantage_stream]
                                       V(s)          A(s,a)
        Q(s,a) = V(s) + A(s,a) - mean(A(s,a))

    Parameters
    ----------
    input_dim:
        Flattened observation size.
    n_actions:
        Number of discrete actions.
    shared_hidden:
        Width of the shared backbone layers.
    stream_hidden:
        Width of the value / advantage stream hidden layers.
    activation:
        ``"relu"`` or ``"tanh"``.
    """

    def __init__(
        self,
        input_dim: int,
        n_actions: int,
        shared_hidden: Tuple[int, ...] = (256,),
        stream_hidden: int = 128,
        activation: str = "relu",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.n_actions = n_actions

        act_fn = nn.ReLU if activation == "relu" else nn.Tanh

        # ---- shared backbone ----
        backbone_layers = []
        in_dim = input_dim
        for h in shared_hidden:
            backbone_layers.append(_orthogonal_init(nn.Linear(in_dim, h)))
            backbone_layers.append(act_fn())
            in_dim = h
        self.backbone = nn.Sequential(*backbone_layers)
        feature_dim = in_dim

        # ---- value stream V(s): scalar ----
        self.value_stream = nn.Sequential(
            _orthogonal_init(nn.Linear(feature_dim, stream_hidden)),
            act_fn(),
            _orthogonal_init(nn.Linear(stream_hidden, 1)),
        )

        # ---- advantage stream A(s,a) ----
        self.advantage_stream = nn.Sequential(
            _orthogonal_init(nn.Linear(feature_dim, stream_hidden)),
            act_fn(),
            _orthogonal_init(nn.Linear(stream_hidden, n_actions)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return Q-values of shape (batch, n_actions)."""
        x = x.float().flatten(start_dim=1) if x.dim() > 2 else x.float()
        features = self.backbone(x)
        V = self.value_stream(features)             # (B, 1)
        A = self.advantage_stream(features)         # (B, nA)
        # Dueling combination: subtract mean advantage for identifiability
        Q = V + A - A.mean(dim=1, keepdim=True)
        return Q

    def q_values(self, obs: torch.Tensor) -> torch.Tensor:
        """Alias of ``forward``."""
        return self.forward(obs)
