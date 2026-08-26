import time

import numpy as np
import torch

import inverse_pinn_training as ipt
from inverse_pinn_model import InversePINN


def train_one_run_diff_lr(
    run_number,
    lr_network,
    lr_r,
    lr_sigma,
    print_every=5000,
    checkpoint_prefix="european_inverse_diff_lr_checkpoint",
    seed=None,
):
    start_time = time.time()

    # Seed: explicit seed for reproducibility, otherwise fresh random seed.
    if seed is None:
        run_seed = torch.seed()
    else:
        run_seed = int(seed)
        torch.manual_seed(run_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(run_seed)

    # All 2000 synthetic observations.
    t_obs, S_obs, V_obs = ipt.load_synthetic_data()
    t_train = t_obs
    S_train = S_obs
    u_train = V_obs / ipt.K

    # Fresh model.
    model = InversePINN(
        t_min=ipt.t_min,
        t_max=ipt.t_max,
        S_min=ipt.S_min,
        S_max=ipt.S_max,
        hidden_width=ipt.hidden_width,
        hidden_layers=ipt.hidden_layers,
        r_initial=ipt.r_initial,
        sigma_initial=ipt.sigma_initial,
    ).to(device=ipt.device, dtype=ipt.dtype)

    # Fresh PDE collocation points.
    t_f = ipt.t_min + ipt.epsilon + (
        ipt.t_max - ipt.t_min - 2.0 * ipt.epsilon
    ) * torch.rand(ipt.N_f, 1, dtype=ipt.dtype, device=ipt.device)

    S_f = ipt.S_min + ipt.epsilon + (
        ipt.S_max - ipt.S_min - 2.0 * ipt.epsilon
    ) * torch.rand(ipt.N_f, 1, dtype=ipt.dtype, device=ipt.device)

    # Fresh terminal points.
    t_terminal = torch.full(
        (ipt.N_terminal, 1),
        fill_value=ipt.T,
        dtype=ipt.dtype,
        device=ipt.device,
    )

    S_terminal = ipt.S_min + (ipt.S_max - ipt.S_min) * torch.rand(
        ipt.N_terminal, 1, dtype=ipt.dtype, device=ipt.device
    )

    # Fresh boundary points.
    t_boundary = ipt.t_min + (ipt.t_max - ipt.t_min) * torch.rand(
        ipt.N_boundary, 1, dtype=ipt.dtype, device=ipt.device
    )

    S_boundary_left = torch.full(
        (ipt.N_boundary, 1),
        fill_value=ipt.S_min,
        dtype=ipt.dtype,
        device=ipt.device,
    )

    S_boundary_right = torch.full(
        (ipt.N_boundary, 1),
        fill_value=ipt.S_max,
        dtype=ipt.dtype,
        device=ipt.device,
    )

    # Separate learning rates.
    optimizer = torch.optim.Adam([
        {"params": model.model.parameters(), "lr": lr_network},
        {"params": [model.r_raw], "lr": lr_r},
        {"params": [model.sigma_raw], "lr": lr_sigma},
    ])

    checkpoint_path = f"{checkpoint_prefix}_run{run_number}.pt"

    print()
    print(f"Starting Run {run_number}")
    print(f"Seed: {run_seed}")
    print(f"Training observations: {t_train.shape[0]}")
    print(
        f"Physics points: PDE={ipt.N_f}, "
        f"terminal={ipt.N_terminal}, boundary={ipt.N_boundary}"
    )
    print(
        f"Learning rates: network={lr_network}, "
        f"r={lr_r}, sigma={lr_sigma}"
    )
    print(
        f"Initial parameters: r={model.r.item():.6f}, "
        f"sigma={model.sigma.item():.6f}"
    )

    for epoch in range(1, ipt.num_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        losses = ipt.compute_losses(
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
            S_boundary_right=S_boundary_right,
        )

        losses["total"].backward()
        optimizer.step()

        if epoch == 1 or epoch % print_every == 0 or epoch == ipt.num_epochs:
            print(
                f"Epoch {epoch:5d} | "
                f"Total: {losses['total'].item():.6e} | "
                f"r: {model.r.item():.6f} | "
                f"sigma: {model.sigma.item():.6f}"
            )

        if epoch % 5000 == 0:
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "run": run_number,
                    "seed": run_seed,
                    "epoch": epoch,
                    "lr_network": lr_network,
                    "lr_r": lr_r,
                    "lr_sigma": lr_sigma,
                    "r_estimate_partial": model.r.item(),
                    "sigma_estimate_partial": model.sigma.item(),
                },
                checkpoint_path,
            )

    r_estimated = model.r.item()
    sigma_estimated = model.sigma.item()

    r_error_percent = abs(r_estimated - ipt.r_true) / abs(ipt.r_true) * 100.0
    sigma_error_percent = (
        abs(sigma_estimated - ipt.sigma_true) / abs(ipt.sigma_true) * 100.0
    )

    # Same deterministic 100 x 100 uniform test mesh.
    N_test = 100

    t_test_1d = torch.linspace(
        ipt.t_min, ipt.t_max, N_test, dtype=ipt.dtype, device=ipt.device
    )
    S_test_1d = torch.linspace(
        ipt.S_min, ipt.S_max, N_test, dtype=ipt.dtype, device=ipt.device
    )

    t_test_mesh, S_test_mesh = torch.meshgrid(
        t_test_1d, S_test_1d, indexing="ij"
    )
    t_test = t_test_mesh.reshape(-1, 1)
    S_test = S_test_mesh.reshape(-1, 1)

    model.eval()
    with torch.no_grad():
        V_true = ipt.black_scholes_call(
            t=t_test,
            S=S_test,
            K=ipt.K,
            T=ipt.T,
            r=ipt.r_true,
            sigma=ipt.sigma_true,
        )
        u_pinn = model(t_test, S_test)
        V_pinn = ipt.K * u_pinn

    price_test_mse = torch.mean((V_true - V_pinn) ** 2).item()

    V_true_np = V_true.cpu().numpy().reshape(-1)
    V_pinn_np = V_pinn.cpu().numpy().reshape(-1)
    price_test_correlation = float(
        np.corrcoef(V_true_np, V_pinn_np)[0, 1]
    )

    runtime_minutes = (time.time() - start_time) / 60.0

    result = {
        "run": run_number,
        "seed": run_seed,
        "lr_network": lr_network,
        "lr_r": lr_r,
        "lr_sigma": lr_sigma,
        "r_estimate": r_estimated,
        "sigma_estimate": sigma_estimated,
        "r_error_percent": r_error_percent,
        "sigma_error_percent": sigma_error_percent,
        "price_mse": price_test_mse,
        "price_correlation": price_test_correlation,
        "runtime_minutes": runtime_minutes,
    }

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "result": result,
            "run": run_number,
            "seed": run_seed,
            "epoch": ipt.num_epochs,
            "lr_network": lr_network,
            "lr_r": lr_r,
            "lr_sigma": lr_sigma,
            "K": ipt.K,
            "T": ipt.T,
            "r_true": ipt.r_true,
            "sigma_true": ipt.sigma_true,
            "r_initial": ipt.r_initial,
            "sigma_initial": ipt.sigma_initial,
            "hidden_width": ipt.hidden_width,
            "hidden_layers": ipt.hidden_layers,
            "N_f": ipt.N_f,
            "N_terminal": ipt.N_terminal,
            "N_boundary": ipt.N_boundary,
            "num_epochs": ipt.num_epochs,
            "N_test": N_test,
        },
        checkpoint_path,
    )

    print()
    print(f"Run {run_number} completed")
    print(f"Estimated r:            {r_estimated:.8f}")
    print(f"Estimated sigma:        {sigma_estimated:.8f}")
    print(f"r relative error:       {r_error_percent:.4f}%")
    print(f"sigma relative error:   {sigma_error_percent:.4f}%")
    print(f"Price test MSE:         {price_test_mse:.8f}")
    print(f"Price test correlation: {price_test_correlation:.8f}")
    print(f"Runtime:                {runtime_minutes:.2f} minutes")
    print(f"Saved: {checkpoint_path}")

    return result
