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
    K, T: fixed scalars 
    r, sigma: torch tensors/parameters (trainable)
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

# given u(t, S, cal ut, uS, uSS and residual
def black_scholes_residual_with_u_SS(model: nn.Module, t: torch.Tensor, S: torch.Tensor):
    """
    Same as inverse_pinn_training.black_scholes_residual, but returns
    the 5-tuple (residual, u, u_t, u_S, u_SS) instead of just residual.
    """
    t_input = t.clone().detach().requires_grad_(True)
    S_input = S.clone().detach().requires_grad_(True)

    u = model(t_input, S_input)

    u_t = torch.autograd.grad(
        outputs=u, inputs=t_input,
        grad_outputs=torch.ones_like(u),
        create_graph=True, retain_graph=True
    )[0]

    u_S = torch.autograd.grad(
        outputs=u, inputs=S_input,
        grad_outputs=torch.ones_like(u),
        create_graph=True, retain_graph=True
    )[0]

    u_SS = torch.autograd.grad(
        outputs=u_S, inputs=S_input,
        grad_outputs=torch.ones_like(u_S),
        create_graph=True, retain_graph=True
    )[0]

    r_est = model.r
    sigma_est = model.sigma

    residual = (
        u_t
        + 0.5 * (sigma_est ** 2) * (S_input ** 2) * u_SS
        + r_est * S_input * u_S
        - r_est * u
    )

    return residual, u, u_t, u_S, u_SS

# regression baseline
def calibrate_bs_regression(
    S_obs, t_obs, V_obs, K, T,
    r_init=0.05, sigma_init=0.20,   # initial guess 
    lr=1e-2, num_epochs=5000, dtype=torch.float64, device="cpu"
):
    """
    Minimize mean((V_market - BS_call(S,t,r,sigma))^2) over (r, sigma)
    using gradient descent. Returns (r_hat, sigma_hat, loss_history).
    """
    r = torch.tensor(r_init, dtype=dtype, device=device, requires_grad=True)
    sigma = torch.tensor(sigma_init, dtype=dtype, device=device, requires_grad=True)

    optimizer = torch.optim.Adam([r, sigma], lr=lr)
    loss_history = []

    for epoch in range(num_epochs):
        optimizer.zero_grad(set_to_none=True)

        V_pred = black_scholes_call(S_obs, t_obs, K, T, r, sigma)
        loss = torch.mean((V_pred - V_obs) ** 2)

        loss.backward()
        optimizer.step()

        # keep sigma positive without softplus (regression side, not the PINN)
        with torch.no_grad():
            sigma.clamp_(min=1e-4)

        loss_history.append(loss.item())

    return r.item(), sigma.item(), loss_history