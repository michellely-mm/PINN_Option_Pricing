import time

import numpy as np
import torch
import torch.nn as nn

from forward_pinn_model import ForwardPINN


device = torch.device("cpu")
dtype = torch.float64



K = 40.0
T = 1.0

t_min = 0.0
t_max = T

S_min = 0.0
S_max = 160.0

# In the FORWARD problem, r and sigma are KNOWN and FIXED.
r_true = 0.05
sigma_true = 0.20



N_f = 2000
N_terminal = 500
N_boundary = 500

epsilon = 1e-4

hidden_width = 64
hidden_layers = 4



learning_rate = 1e-3
num_epochs = 30000

# No data loss is used in the forward PINN.
lambda_pde = 1.50
lambda_terminal = 1.0
lambda_boundary = 1.0



def black_scholes_residual(
    model: nn.Module,
    t: torch.Tensor,
    S: torch.Tensor
):
    # Create differentiable copies of the collocation inputs
    t_input = (
        t.clone()
        .detach()
        .requires_grad_(True)
    )

    S_input = (
        S.clone()
        .detach()
        .requires_grad_(True)
    )

    # Model predicts normalized price u = V / K
    u = model(
        t_input,
        S_input
    )

    # First derivative with respect to time
    u_t = torch.autograd.grad(
        outputs=u,
        inputs=t_input,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True
    )[0]

    # First derivative with respect to spot
    u_S = torch.autograd.grad(
        outputs=u,
        inputs=S_input,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True
    )[0]

    # Second derivative with respect to spot
    u_SS = torch.autograd.grad(
        outputs=u_S,
        inputs=S_input,
        grad_outputs=torch.ones_like(u_S),
        create_graph=True,
        retain_graph=True
    )[0]

    # r and sigma are fixed buffers in ForwardPINN
    r = model.r
    sigma = model.sigma

    residual = (
        u_t

        + 0.5
        * sigma**2
        * S_input**2
        * u_SS

        + r
        * S_input
        * u_S

        - r
        * u
    )

    return residual


# 6. Forward-PINN loss

def compute_losses(
    model,
    t_f,
    S_f,
    t_terminal,
    S_terminal,
    t_boundary,
    S_boundary_left,
    S_boundary_right
):
    pde_prediction = black_scholes_residual(
        model,
        t_f,
        S_f
    )

    pde_true = torch.zeros_like(
        pde_prediction
    )

    loss_pde = torch.mean(
        (
            pde_true
            - pde_prediction
        ) ** 2
    )

    # --------------------------------------------------------
    # Terminal-condition loss
    #
    # V(T,S) = max(S-K,0)
    # u(T,S) = max(S-K,0) / K
    # --------------------------------------------------------

    u_terminal_prediction = model(
        t_terminal,
        S_terminal
    )

    u_terminal_true = (
        torch.relu(
            S_terminal - K
        )
        / K
    )

    loss_terminal = torch.mean(
        (
            u_terminal_true
            - u_terminal_prediction
        ) ** 2
    )


    u_left_prediction = model(
        t_boundary,
        S_boundary_left
    )

    u_left_true = torch.zeros_like(
        u_left_prediction
    )

    loss_boundary_left = torch.mean(
        (
            u_left_true
            - u_left_prediction
        ) ** 2
    )



    u_right_prediction = model(
        t_boundary,
        S_boundary_right
    )

    u_right_true = (
        S_boundary_right

        - K
        * torch.exp(
            -model.r
            * (T - t_boundary)
        )
    ) / K

    loss_boundary_right = torch.mean(
        (
            u_right_true
            - u_right_prediction
        ) ** 2
    )

    loss_boundary = (
        loss_boundary_left
        + loss_boundary_right
    )

    total_loss = (
        lambda_pde
        * loss_pde

        + lambda_terminal
        * loss_terminal

        + lambda_boundary
        * loss_boundary
    )

    return {
        "total": total_loss,
        "pde": loss_pde,
        "terminal": loss_terminal,
        "boundary": loss_boundary
    }



def black_scholes_call(
    t,
    S,
    K,
    T,
    r,
    sigma
):
    tau = T - t

    eps = 1e-12

    tau_safe = torch.clamp(
        tau,
        min=eps
    )

    S_safe = torch.clamp(
        S,
        min=eps
    )

    d1 = (
        torch.log(
            S_safe / K
        )

        + (
            r
            + 0.5 * sigma**2
        )
        * tau_safe

    ) / (
        sigma
        * torch.sqrt(
            tau_safe
        )
    )

    d2 = (
        d1
        - sigma
        * torch.sqrt(
            tau_safe
        )
    )

    N_d1 = 0.5 * (
        1.0
        + torch.erf(
            d1 / np.sqrt(2.0)
        )
    )

    N_d2 = 0.5 * (
        1.0
        + torch.erf(
            d2 / np.sqrt(2.0)
        )
    )

    V = (
        S_safe
        * N_d1

        - K
        * torch.exp(
            -r * tau_safe
        )
        * N_d2
    )

    # Exact terminal condition
    payoff = torch.relu(
        S - K
    )

    V = torch.where(
        tau <= eps,
        payoff,
        V
    )

    # Exact left boundary
    V = torch.where(
        S <= eps,
        torch.zeros_like(V),
        V
    )

    return V


def train_one_run(
    run_number=None,
    print_every=5000,
    return_model=False,
    track_history=False,
    seed=None
):
    start_time = time.time()

    # Optional reproducibility
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)


    model = ForwardPINN(
        t_min=t_min,
        t_max=t_max,
        S_min=S_min,
        S_max=S_max,
        r=r_true,
        sigma=sigma_true,
        hidden_width=hidden_width,
        hidden_layers=hidden_layers
    ).to(
        device=device,
        dtype=dtype
    )


    t_f = (
        t_min
        + epsilon
        + (
            t_max
            - t_min
            - 2.0 * epsilon
        )
        * torch.rand(
            N_f,
            1,
            dtype=dtype,
            device=device
        )
    )

    S_f = (
        S_min
        + epsilon
        + (
            S_max
            - S_min
            - 2.0 * epsilon
        )
        * torch.rand(
            N_f,
            1,
            dtype=dtype,
            device=device
        )
    )


    t_terminal = torch.full(
        (N_terminal, 1),
        fill_value=T,
        dtype=dtype,
        device=device
    )

    S_terminal = (
        S_min
        + (
            S_max - S_min
        )
        * torch.rand(
            N_terminal,
            1,
            dtype=dtype,
            device=device
        )
    )


    t_boundary = (
        t_min
        + (
            t_max - t_min
        )
        * torch.rand(
            N_boundary,
            1,
            dtype=dtype,
            device=device
        )
    )

    S_boundary_left = torch.full(
        (N_boundary, 1),
        fill_value=S_min,
        dtype=dtype,
        device=device
    )

    S_boundary_right = torch.full(
        (N_boundary, 1),
        fill_value=S_max,
        dtype=dtype,
        device=device
    )


    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )


    history = None

    if track_history:
        history = {
            "epoch": [],
            "total_loss": [],
            "pde_loss": [],
            "terminal_loss": [],
            "boundary_loss": []
        }


    if run_number is not None:
        print()
        print(
            f"Starting Forward PINN Run {run_number}"
        )

    print(
        f"Fixed r: {model.r.item():.6f}"
    )
    print(
        f"Fixed sigma: {model.sigma.item():.6f}"
    )

    for epoch in range(
        1,
        num_epochs + 1
    ):
        model.train()

        optimizer.zero_grad(
            set_to_none=True
        )

        losses = compute_losses(
            model=model,
            t_f=t_f,
            S_f=S_f,
            t_terminal=t_terminal,
            S_terminal=S_terminal,
            t_boundary=t_boundary,
            S_boundary_left=S_boundary_left,
            S_boundary_right=S_boundary_right
        )

        losses["total"].backward()

        # Updates only NN weights
        optimizer.step()

        if track_history:
            history["epoch"].append(
                epoch
            )

            history["total_loss"].append(
                losses["total"].item()
            )

            history["pde_loss"].append(
                losses["pde"].item()
            )

            history["terminal_loss"].append(
                losses["terminal"].item()
            )

            history["boundary_loss"].append(
                losses["boundary"].item()
            )

        if (
            epoch == 1
            or epoch % print_every == 0
            or epoch == num_epochs
        ):
            print(
                f"Epoch {epoch:5d} | "
                f"Total: {losses['total'].item():.6e} | "
                f"PDE: {losses['pde'].item():.6e} | "
                f"Terminal: {losses['terminal'].item():.6e} | "
                f"Boundary: {losses['boundary'].item():.6e}"
            )

   

    N_test = 100

    t_test_1d = torch.linspace(
        t_min,
        t_max,
        N_test,
        dtype=dtype,
        device=device
    )

    S_test_1d = torch.linspace(
        S_min,
        S_max,
        N_test,
        dtype=dtype,
        device=device
    )

    t_test_mesh, S_test_mesh = torch.meshgrid(
        t_test_1d,
        S_test_1d,
        indexing="ij"
    )

    t_test = (
        t_test_mesh
        .reshape(-1, 1)
    )

    S_test = (
        S_test_mesh
        .reshape(-1, 1)
    )

    model.eval()

    with torch.no_grad():
        V_true = black_scholes_call(
            t=t_test,
            S=S_test,
            K=K,
            T=T,
            r=r_true,
            sigma=sigma_true
        )

        u_pinn = model(
            t_test,
            S_test
        )

        V_pinn = (
            K * u_pinn
        )

    # This is PRICE / SOLUTION MSE on the test mesh.
    price_test_mse = torch.mean(
        (
            V_true
            - V_pinn
        ) ** 2
    ).item()

    V_true_np = (
        V_true
        .cpu()
        .numpy()
        .reshape(-1)
    )

    V_pinn_np = (
        V_pinn
        .cpu()
        .numpy()
        .reshape(-1)
    )

    price_test_correlation = np.corrcoef(
        V_true_np,
        V_pinn_np
    )[0, 1]

    runtime_minutes = (
        time.time()
        - start_time
    ) / 60.0

    print()
    print(
        f"Forward PINN Run {run_number} completed"
        if run_number is not None
        else "Forward PINN training completed"
    )

    print(
        f"Fixed r:                "
        f"{model.r.item():.8f}"
    )

    print(
        f"Fixed sigma:            "
        f"{model.sigma.item():.8f}"
    )

    print(
        f"Price test MSE:         "
        f"{price_test_mse:.8f}"
    )

    print(
        f"Price test correlation: "
        f"{price_test_correlation:.8f}"
    )

    print(
        f"Runtime:                "
        f"{runtime_minutes:.2f} minutes"
    )

    result = {
        "run": run_number,
        "r_fixed": model.r.item(),
        "sigma_fixed": model.sigma.item(),
        "price_test_mse": price_test_mse,
        "price_test_correlation": price_test_correlation,
        "runtime_minutes": runtime_minutes
    }

    if return_model and track_history:
        return (
            result,
            model,
            history
        )

    elif return_model:
        return (
            result,
            model
        )

    elif track_history:
        return (
            result,
            history
        )

    else:
        return result
