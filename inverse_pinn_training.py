from pathlib import Path
import time

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from inverse_pinn_model import InversePINN


# 1. Numerical settings

device = torch.device("cpu")
dtype = torch.float64


# 2. European call problem settings

K = 40.0
T = 1.0

t_min = 0.0
t_max = T

S_min = 0.0
S_max = 160.0

r_true = 0.05
sigma_true = 0.20


# 3. PINN settings

N_f = 2000
N_terminal = 500
N_boundary = 500

epsilon = 1e-4

hidden_width = 64
hidden_layers = 4

r_initial = 0.10
sigma_initial = 0.40


# 4. Training settings

learning_rate = 1e-3
num_epochs = 30000

lambda_data = 6.0
lambda_pde = 1.50
lambda_terminal = 1.0
lambda_boundary = 1.0


# 5. Load synthetic observations



def load_synthetic_data(csv_path=None):
    """
    Load the fixed synthetic European-call observations from CSV.

    The synthetic observations are NOT regenerated here.

    Expected CSV:
        synthetic_european_call_data_2000.csv

    Expected columns:
        t, S, V

    The CSV is located automatically so the code does not
    depend on an absolute local machine path.
    """

    from pathlib import Path
    import pandas as pd
    import torch


    filename = (
        "synthetic_european_call_data_2000.csv"
    )


    if csv_path is not None:

        csv_path = (
            Path(
                csv_path
            )
            .expanduser()
            .resolve()
        )

        if not csv_path.is_file():

            raise FileNotFoundError(
                f"CSV file does not exist: {csv_path}"
            )


    else:

        module_dir = (
            Path(__file__)
            .resolve()
            .parent
        )

        working_dir = (
            Path.cwd()
            .resolve()
        )


        candidates = [

            module_dir
            / filename,

            module_dir
            / "data"
            / filename,

            working_dir
            / filename,

            working_dir
            / "data"
            / filename,

            module_dir
            .parent
            / filename,

            module_dir
            .parent
            / "data"
            / filename,
        ]


        csv_path = None


        for candidate in candidates:

            if candidate.is_file():

                csv_path = (
                    candidate
                    .resolve()
                )

                break


        if csv_path is None:

            search_roots = [
                module_dir,
                working_dir
            ]


            seen = set()


            for root in search_roots:

                root = root.resolve()

                if root in seen:
                    continue

                seen.add(
                    root
                )


                try:

                    matches = list(
                        root.rglob(
                            filename
                        )
                    )

                except Exception:

                    matches = []


                if matches:

                    csv_path = (
                        matches[0]
                        .resolve()
                    )

                    break


        if csv_path is None:

            raise FileNotFoundError(
                "\nCould not find "
                "synthetic_european_call_data_2000.csv.\n"
                "Make sure the CSV is included somewhere "
                "inside the downloaded project folder.\n"
            )


    synthetic_df = pd.read_csv(
        csv_path
    )


    required_columns = {
        "t",
        "S",
        "V"
    }


    missing_columns = (
        required_columns
        -
        set(
            synthetic_df.columns
        )
    )


    if missing_columns:

        raise ValueError(
            "Synthetic CSV is missing required columns: "
            f"{sorted(missing_columns)}"
        )


    t_obs = torch.tensor(
        synthetic_df[
            "t"
        ].to_numpy(),
        dtype=dtype,
        device=device
    ).reshape(
        -1,
        1
    )


    S_obs = torch.tensor(
        synthetic_df[
            "S"
        ].to_numpy(),
        dtype=dtype,
        device=device
    ).reshape(
        -1,
        1
    )


    V_obs = torch.tensor(
        synthetic_df[
            "V"
        ].to_numpy(),
        dtype=dtype,
        device=device
    ).reshape(
        -1,
        1
    )


    if not torch.isfinite(
        t_obs
    ).all():

        raise ValueError(
            "Non-finite values found in t."
        )


    if not torch.isfinite(
        S_obs
    ).all():

        raise ValueError(
            "Non-finite values found in S."
        )


    if not torch.isfinite(
        V_obs
    ).all():

        raise ValueError(
            "Non-finite values found in V."
        )


    print()
    print(
        "=" * 60
    )

    print(
        "Loaded fixed synthetic observations from:"
    )

    print(
        csv_path
    )

    print(
        "Number of observations:",
        len(
            synthetic_df
        )
    )

    print(
        "=" * 60
    )

    print()


    return (
        t_obs,
        S_obs,
        V_obs
    )


def black_scholes_residual(
    model: nn.Module,
    t: torch.Tensor,
    S: torch.Tensor
):

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

    u = model(
        t_input,
        S_input
    )

    u_t = torch.autograd.grad(
        outputs=u,
        inputs=t_input,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True
    )[0]

    u_S = torch.autograd.grad(
        outputs=u,
        inputs=S_input,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True
    )[0]

    u_SS = torch.autograd.grad(
        outputs=u_S,
        inputs=S_input,
        grad_outputs=torch.ones_like(u_S),
        create_graph=True,
        retain_graph=True
    )[0]

    r_est = model.r
    sigma_est = model.sigma

    residual = (
        u_t

        + 0.5
        * sigma_est**2
        * S_input**2
        * u_SS

        + r_est
        * S_input
        * u_S

        - r_est
        * u
    )

    return residual


# 7. Loss function

def compute_losses(
    model,
    t_train,
    S_train,
    u_train,
    t_f,
    S_f,
    t_terminal,
    S_terminal,
    t_boundary,
    S_boundary_left,
    S_boundary_right
):

    # Data loss

    u_data_prediction = model(
        t_train,
        S_train
    )

    loss_data = torch.mean(
        (
            u_data_prediction
            - u_train
        ) ** 2
    )


    # PDE loss

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


    # Terminal-condition loss

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


    # Left-boundary loss

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


    # Right-boundary loss

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
        lambda_data
        * loss_data

        + lambda_pde
        * loss_pde

        + lambda_terminal
        * loss_terminal

        + lambda_boundary
        * loss_boundary
    )


    return {
        "total": total_loss,
        "data": loss_data,
        "pde": loss_pde,
        "terminal": loss_terminal,
        "boundary": loss_boundary
    }


# 8. Analytical Black-Scholes European call solution

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


    payoff = torch.relu(
        S - K
    )


    V = torch.where(
        tau <= eps,
        payoff,
        V
    )


    V = torch.where(
        S <= eps,
        torch.zeros_like(V),
        V
    )


    return V


# 9. Train one independent inverse-PINN run

def train_one_run(
    run_number=None,
    print_every=5000,
    return_model=False,
    track_history=False,
    track_adam_updates=False,
    data_path=None
):

    start_time = time.time()


    # Load observations

    t_obs, S_obs, V_obs = (
        load_synthetic_data(
            data_path=data_path
        )
    )

    N_obs = t_obs.shape[0]


    # Use all observations for training

    t_train = t_obs
    S_train = S_obs
    V_train = V_obs


    # Normalize option prices

    u_train = (
        V_train / K
    )


    # Create fresh model

    model = InversePINN(
        t_min=t_min,
        t_max=t_max,
        S_min=S_min,
        S_max=S_max,
        hidden_width=hidden_width,
        hidden_layers=hidden_layers,
        r_initial=r_initial,
        sigma_initial=sigma_initial
    ).to(
        device=device,
        dtype=dtype
    )


    # Generate PDE collocation points

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


    # Generate terminal-condition points

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


    # Generate boundary-condition points

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


    # Create optimizer

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )


    # Optional regular training history

    history = None

    if track_history:

        history = {
            "epoch": [],
            "total_loss": [],
            "data_loss": [],
            "pde_loss": [],
            "terminal_loss": [],
            "boundary_loss": [],
            "r_estimate": [],
            "sigma_estimate": []
        }


    # Optional gradient and Adam-update history

    adam_update_history = None

    if track_adam_updates:

        adam_update_history = {
            "epoch": [],
            "r_raw_grad": [],
            "sigma_raw_grad": [],
            "r_raw_update": [],
            "sigma_raw_update": [],
            "r_update": [],
            "sigma_update": []
        }


    # Print starting information

    if run_number is not None:

        print()

        print(
            f"Starting Run {run_number}"
        )

        print(
            f"Training observations: {N_obs}"
        )


    # Training loop

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

            t_train=t_train,
            S_train=S_train,
            u_train=u_train,

            t_f=t_f,
            S_f=S_f,

            t_terminal=t_terminal,
            S_terminal=S_terminal,

            t_boundary=t_boundary,
            S_boundary_left=S_boundary_left,
            S_boundary_right=S_boundary_right
        )


        # Backpropagation

        losses["total"].backward()


        # Record gradients and parameter values before Adam step

        if track_adam_updates:

            r_raw_grad = (
                model.r_raw.grad.item()
            )

            sigma_raw_grad = (
                model.sigma_raw.grad.item()
            )

            r_raw_before = (
                model.r_raw
                .detach()
                .item()
            )

            sigma_raw_before = (
                model.sigma_raw
                .detach()
                .item()
            )

            r_before = (
                model.r
                .detach()
                .item()
            )

            sigma_before = (
                model.sigma
                .detach()
                .item()
            )


        # Update neural-network weights, r_raw, and sigma_raw

        optimizer.step()


        # Record actual Adam updates

        if track_adam_updates:

            r_raw_after = (
                model.r_raw
                .detach()
                .item()
            )

            sigma_raw_after = (
                model.sigma_raw
                .detach()
                .item()
            )

            r_after = (
                model.r
                .detach()
                .item()
            )

            sigma_after = (
                model.sigma
                .detach()
                .item()
            )


            adam_update_history[
                "epoch"
            ].append(
                epoch
            )


            adam_update_history[
                "r_raw_grad"
            ].append(
                r_raw_grad
            )


            adam_update_history[
                "sigma_raw_grad"
            ].append(
                sigma_raw_grad
            )


            adam_update_history[
                "r_raw_update"
            ].append(
                r_raw_after
                - r_raw_before
            )


            adam_update_history[
                "sigma_raw_update"
            ].append(
                sigma_raw_after
                - sigma_raw_before
            )


            adam_update_history[
                "r_update"
            ].append(
                r_after
                - r_before
            )


            adam_update_history[
                "sigma_update"
            ].append(
                sigma_after
                - sigma_before
            )


        # Save regular training history

        if track_history:

            history[
                "epoch"
            ].append(
                epoch
            )

            history[
                "total_loss"
            ].append(
                losses["total"].item()
            )

            history[
                "data_loss"
            ].append(
                losses["data"].item()
            )

            history[
                "pde_loss"
            ].append(
                losses["pde"].item()
            )

            history[
                "terminal_loss"
            ].append(
                losses["terminal"].item()
            )

            history[
                "boundary_loss"
            ].append(
                losses["boundary"].item()
            )

            history[
                "r_estimate"
            ].append(
                model.r.item()
            )

            history[
                "sigma_estimate"
            ].append(
                model.sigma.item()
            )


        # Print training progress

        if (
            epoch == 1
            or epoch % print_every == 0
            or epoch == num_epochs
        ):

            print(
                f"Epoch {epoch:5d} | "
                f"Total: "
                f"{losses['total'].item():.6e} | "
                f"r: "
                f"{model.r.item():.6f} | "
                f"sigma: "
                f"{model.sigma.item():.6f}"
            )


    # Final parameter estimates

    r_estimated = (
        model.r.item()
    )

    sigma_estimated = (
        model.sigma.item()
    )


    # Parameter-estimation errors

    r_absolute_error = abs(
        r_estimated
        - r_true
    )

    sigma_absolute_error = abs(
        sigma_estimated
        - sigma_true
    )


    r_error_percent = (
        r_absolute_error
        / abs(r_true)
        * 100.0
    )


    sigma_error_percent = (
        sigma_absolute_error
        / abs(sigma_true)
        * 100.0
    )


    # Build fixed 100 x 100 PDE test mesh

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


    t_test_mesh, S_test_mesh = (
        torch.meshgrid(
            t_test_1d,
            S_test_1d,
            indexing="ij"
        )
    )


    t_test = (
        t_test_mesh
        .reshape(-1, 1)
    )


    S_test = (
        S_test_mesh
        .reshape(-1, 1)
    )


    # Evaluate analytical solution and PINN prediction

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
            K
            * u_pinn
        )


    # PDE test MSE

    pde_test_mse = torch.mean(
        (
            V_true
            - V_pinn
        ) ** 2
    ).item()


    # PDE test correlation

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


    pde_test_correlation = np.corrcoef(
        V_true_np,
        V_pinn_np
    )[0, 1]


    # Runtime

    runtime_minutes = (
        time.time()
        - start_time
    ) / 60.0


    # Print final results

    print()

    print(
        f"Run {run_number} completed"
        if run_number is not None
        else "Training completed"
    )


    print(
        f"Estimated r:           "
        f"{r_estimated:.8f}"
    )


    print(
        f"Estimated sigma:       "
        f"{sigma_estimated:.8f}"
    )


    print(
        f"r relative error:      "
        f"{r_error_percent:.4f}%"
    )


    print(
        f"sigma relative error:  "
        f"{sigma_error_percent:.4f}%"
    )


    print(
        f"PDE test MSE:          "
        f"{pde_test_mse:.8f}"
    )


    print(
        f"PDE test correlation:  "
        f"{pde_test_correlation:.8f}"
    )


    print(
        f"Runtime:               "
        f"{runtime_minutes:.2f} minutes"
    )


    # Results dictionary

    result = {

        "run":
            run_number,

        "r_estimate":
            r_estimated,

        "sigma_estimate":
            sigma_estimated,

        "r_error_percent":
            r_error_percent,

        "sigma_error_percent":
            sigma_error_percent,

        "pde_mse":
            pde_test_mse,

        "pde_correlation":
            pde_test_correlation,

        "runtime_minutes":
            runtime_minutes
    }


    # Return requested outputs

    if (
        return_model
        and track_history
        and track_adam_updates
    ):

        return (
            result,
            model,
            history,
            adam_update_history
        )


    elif (
        return_model
        and track_history
    ):

        return (
            result,
            model,
            history
        )


    elif (
        return_model
        and track_adam_updates
    ):

        return (
            result,
            model,
            adam_update_history
        )


    elif (
        track_history
        and track_adam_updates
    ):

        return (
            result,
            history,
            adam_update_history
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


    elif track_adam_updates:

        return (
            result,
            adam_update_history
        )


    else:

        return result