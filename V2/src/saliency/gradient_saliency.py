"""Neural-network input saliency / attribution methods.

Implements four attribution techniques of increasing sophistication:

1. **Vanilla Gradient**   — ∂output/∂input (Simonyan et al., 2013)
2. **Gradient × Input**   — element-wise product of gradient and input
3. **SmoothGrad**         — average gradient over noisy input copies (Smilkov et al., 2017)
4. **Integrated Gradients** — path integral from baseline to input (Sundararajan et al., 2017)

All functions accept a PyTorch network, an observation tensor, and a
target action index, and return a numpy attribution map of the same shape
as the observation.

References
----------
Simonyan et al., 2013. "Deep Inside Convolutional Networks."
Smilkov et al., 2017. "SmoothGrad: removing noise by adding noise."
Sundararajan et al., 2017. "Axiomatic Attribution for Deep Networks."
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from typing import Optional


# ---------------------------------------------------------------------------
# Helper: prepare observation tensor
# ---------------------------------------------------------------------------

def _to_tensor(obs: np.ndarray, device: torch.device) -> torch.Tensor:
    """Convert numpy obs to a (1, D) float tensor with gradient tracking."""
    x = torch.tensor(obs.flatten(), dtype=torch.float32, device=device)
    x = x.unsqueeze(0)  # (1, D)
    x.requires_grad_(True)
    return x


# ---------------------------------------------------------------------------
# 1. Vanilla gradient
# ---------------------------------------------------------------------------

def vanilla_gradient(
    network: nn.Module,
    obs: np.ndarray,
    action: int,
    device: torch.device = torch.device("cpu"),
) -> np.ndarray:
    """Compute ∂Q(s,a)/∂s via backpropagation.

    Parameters
    ----------
    network:
        The Q-network (or actor network) whose output is being attributed.
    obs:
        Input observation as a numpy array (any shape — will be flattened).
    action:
        The action index to differentiate w.r.t.
    device:
        Torch device.

    Returns
    -------
    np.ndarray
        Attribution map of the same shape as *obs*.
    """
    network.eval()
    x = _to_tensor(obs, device)

    output = network(x)
    # Handle both scalar outputs (critic) and vector outputs (Q-net / actor)
    if output.dim() > 1 and output.shape[1] > 1:
        target = output[0, action]
    else:
        target = output[0]

    network.zero_grad()
    target.backward()

    grad = x.grad.detach().cpu().numpy().reshape(obs.shape)
    return grad


# ---------------------------------------------------------------------------
# 2. Gradient × Input
# ---------------------------------------------------------------------------

def gradient_x_input(
    network: nn.Module,
    obs: np.ndarray,
    action: int,
    device: torch.device = torch.device("cpu"),
) -> np.ndarray:
    """Element-wise product of gradient and input: grad(s) ⊙ s.

    Reduces the noise present in raw gradients by down-weighting
    near-zero input features even when their gradients are large.
    """
    grad = vanilla_gradient(network, obs, action, device)
    return grad * obs


# ---------------------------------------------------------------------------
# 3. SmoothGrad
# ---------------------------------------------------------------------------

def smoothgrad(
    network: nn.Module,
    obs: np.ndarray,
    action: int,
    n_samples: int = 50,
    noise_level: float = 0.1,
    device: torch.device = torch.device("cpu"),
) -> np.ndarray:
    """Average gradient over n_samples noisy copies of the input.

    Parameters
    ----------
    n_samples:
        Number of noisy samples to average (higher → smoother but slower).
    noise_level:
        Standard deviation of additive Gaussian noise as a fraction of
        the input range (std = noise_level * (max - min)).

    Returns
    -------
    np.ndarray
        Averaged attribution map of the same shape as *obs*.
    """
    obs_range = float(obs.max() - obs.min()) + 1e-8
    std = noise_level * obs_range

    grads = []
    for _ in range(n_samples):
        noisy_obs = obs + np.random.normal(0, std, obs.shape).astype(np.float32)
        g = vanilla_gradient(network, noisy_obs, action, device)
        grads.append(g)

    return np.mean(grads, axis=0)


# ---------------------------------------------------------------------------
# 4. Integrated Gradients (Sundararajan et al., 2017)
# ---------------------------------------------------------------------------

def integrated_gradients(
    network: nn.Module,
    obs: np.ndarray,
    action: int,
    baseline: Optional[np.ndarray] = None,
    n_steps: int = 50,
    device: torch.device = torch.device("cpu"),
) -> np.ndarray:
    """Compute Integrated Gradients from *baseline* to *obs*.

    IG satisfies the *Completeness* axiom:  the sum of attributions equals
    the difference in network output between the input and the baseline.

    Parameters
    ----------
    baseline:
        Reference input (default: zero tensor — black image / zero features).
    n_steps:
        Number of interpolation steps along the integral path.
        Higher → more accurate approximation of the integral.

    Returns
    -------
    np.ndarray
        Attribution map of the same shape as *obs*, satisfying completeness.
    """
    if baseline is None:
        baseline = np.zeros_like(obs, dtype=np.float32)

    network.eval()
    accumulated_grads = np.zeros_like(obs, dtype=np.float64)

    for step in range(n_steps + 1):
        alpha = step / n_steps
        interpolated = baseline + alpha * (obs - baseline)
        g = vanilla_gradient(network, interpolated, action, device)
        accumulated_grads += g.astype(np.float64)

    # Trapezoidal rule: average gradient × (input - baseline)
    avg_grads = accumulated_grads / (n_steps + 1)
    ig = avg_grads * (obs - baseline).astype(np.float64)

    return ig.astype(np.float32)


# ---------------------------------------------------------------------------
# Completeness check (for debugging / validation)
# ---------------------------------------------------------------------------

def check_completeness(
    network: nn.Module,
    obs: np.ndarray,
    action: int,
    attribution: np.ndarray,
    baseline: Optional[np.ndarray] = None,
    device: torch.device = torch.device("cpu"),
    rtol: float = 0.05,
) -> dict:
    """Verify that sum(attribution) ≈ F(obs) - F(baseline).

    Returns a dict with ``"passed"``, ``"sum_attr"``, ``"delta_F"``, ``"error"``.
    """
    if baseline is None:
        baseline = np.zeros_like(obs, dtype=np.float32)

    def _forward(x):
        xt = torch.tensor(x.flatten(), dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            out = network(xt)
        if out.dim() > 1 and out.shape[1] > 1:
            return float(out[0, action].item())
        return float(out[0].item())

    F_input    = _forward(obs)
    F_baseline = _forward(baseline)
    delta_F    = F_input - F_baseline
    sum_attr   = float(attribution.sum())
    error      = abs(sum_attr - delta_F) / (abs(delta_F) + 1e-8)

    return {
        "passed":   error < rtol,
        "sum_attr": sum_attr,
        "delta_F":  delta_F,
        "error":    error,
    }
