"""
Shared utilities for the real-data (NDX) inverse PINN experiment.
Covers: 
- analytical Black-Scholes pricing
- PDE residual with u_SS exposed
- closed-form regression calibration baseline (paper eq. 5)
"""

import numpy as np
import torch
import torch.nn as nn

def black_scholes_call(S, t, K, T, r, sigma):
    """
    European call price under Black-Scholes.
    S, t: tensors, same convention as the PINN (t=0 today, t=T maturity)
    K, T: fixed scalars (Python floats)
    r, sigma: may be Python floats OR torch tensors/parameters (trainable)
    """
    tau = T - t
    eps = 1e-12
    tau_safe = torch.clamp(tau, min=eps)
    S_safe = torch.clamp(S, min=eps)

    d1 = (
        torch.log(S_safe / K)
        + (r + 0.5 * sigma ** 2) * tau_safe
    ) / (sigma * torch.sqrt(tau_safe))

    d2 = d1 - sigma * torch.sqrt(tau_safe)

    N_d1 = 0.5 * (1.0 + torch.erf(d1 / np.sqrt(2.0)))
    N_d2 = 0.5 * (1.0 + torch.erf(d2 / np.sqrt(2.0)))

    V = S_safe * N_d1 - K * torch.exp(-r * tau_safe) * N_d2

    payoff = torch.relu(S - K)
    V = torch.where(tau <= eps, payoff, V)
    V = torch.where(S <= eps, torch.zeros_like(V), V)

    return V