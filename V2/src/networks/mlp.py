"""Multi-layer perceptron building blocks for deep RL agents.

Provides:
  - MLP          — configurable feed-forward network (Q-head or policy/value head)
  - ActorMLP     — stochastic discrete actor (outputs action logits)
  - CriticMLP    — state-value estimator (scalar output)

All networks use orthogonal weight initialisation (common in deep RL)
and support an optional layer-normalisation after each hidden activation.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Initialisation helper
# ---------------------------------------------------------------------------

def _orthogonal_init(layer: nn.Linear, gain: float = 1.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0.0)
    return layer


# ---------------------------------------------------------------------------
# General-purpose MLP
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """Configurable feed-forward network.

    Parameters
    ----------
    input_dim:
        Flattened observation dimension.
    output_dim:
        Number of output units (e.g. nA for Q-heads, 1 for value heads).
    hidden_sizes:
        Sequence of hidden-layer widths.  Default (256, 256).
    activation:
        ``"relu"`` or ``"tanh"``.
    layer_norm:
        If True, apply LayerNorm after each hidden activation.
    output_gain:
        Orthogonal init gain for the output layer.
        Typical values: 1.0 (Q / value), 0.01 (policy logits).
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_sizes: Tuple[int, ...] = (256, 256),
        activation: str = "relu",
        layer_norm: bool = False,
        output_gain: float = 1.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        act_fn = nn.ReLU if activation == "relu" else nn.Tanh

        layers: List[nn.Module] = []
        in_dim = input_dim
        for h in hidden_sizes:
            layers.append(_orthogonal_init(nn.Linear(in_dim, h)))
            if layer_norm:
                layers.append(nn.LayerNorm(h))
            layers.append(act_fn())
            in_dim = h

        self.backbone = nn.Sequential(*layers)
        self.head = _orthogonal_init(nn.Linear(in_dim, output_dim), gain=output_gain)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float().flatten(start_dim=1) if x.dim() > 2 else x.float()
        return self.head(self.backbone(x))


# ---------------------------------------------------------------------------
# Actor (stochastic discrete policy)
# ---------------------------------------------------------------------------

class ActorMLP(nn.Module):
    """Discrete policy network that outputs action logits.

    Call ``get_dist(obs)`` to obtain a ``Categorical`` distribution,
    or ``forward(obs)`` for raw logits.
    """

    def __init__(
        self,
        input_dim: int,
        n_actions: int,
        hidden_sizes: Tuple[int, ...] = (256, 256),
        activation: str = "tanh",
        layer_norm: bool = False,
    ):
        super().__init__()
        self.net = MLP(
            input_dim, n_actions, hidden_sizes,
            activation=activation, layer_norm=layer_norm,
            output_gain=0.01,  # small gain → near-uniform initial policy
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def get_dist(self, x: torch.Tensor) -> torch.distributions.Categorical:
        logits = self.forward(x)
        return torch.distributions.Categorical(logits=logits)

    def evaluate_actions(
        self, x: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (log_probs, entropy) for the given actions."""
        dist = self.get_dist(x)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy


# ---------------------------------------------------------------------------
# Critic (state-value estimator)
# ---------------------------------------------------------------------------

class CriticMLP(nn.Module):
    """State-value V(s) network — outputs a single scalar per state."""

    def __init__(
        self,
        input_dim: int,
        hidden_sizes: Tuple[int, ...] = (256, 256),
        activation: str = "tanh",
        layer_norm: bool = False,
    ):
        super().__init__()
        self.net = MLP(
            input_dim, 1, hidden_sizes,
            activation=activation, layer_norm=layer_norm,
            output_gain=1.0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)
