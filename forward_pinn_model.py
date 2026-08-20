import torch
import torch.nn as nn


# Keep the same dtype as the inverse PINN
dtype = torch.float64


class ForwardPINN(nn.Module):
    """
    Forward PINN for the European call Black-Scholes problem.

    Inputs:
        t = current time
        S = current spot price

    Output:
        u(t, S) = V(t, S) / K

    Known fixed PDE parameters:
        r
        sigma

    Only the neural-network weights are trainable.
    """

    def __init__(
        self,
        t_min: float,
        t_max: float,
        S_min: float,
        S_max: float,
        r: float = 0.05,
        sigma: float = 0.20,
        hidden_width: int = 64,
        hidden_layers: int = 4
    ):
        super().__init__()

        if hidden_layers < 1:
            raise ValueError("hidden_layers must be at least 1.")

        # Save domain boundaries
        self.register_buffer(
            "t_min_tensor",
            torch.tensor(t_min, dtype=dtype)
        )

        self.register_buffer(
            "t_max_tensor",
            torch.tensor(t_max, dtype=dtype)
        )

        self.register_buffer(
            "S_min_tensor",
            torch.tensor(S_min, dtype=dtype)
        )

        self.register_buffer(
            "S_max_tensor",
            torch.tensor(S_max, dtype=dtype)
        )

        # Known fixed Black-Scholes parameters.
        # They move with the model/device but are NOT trainable.
        self.register_buffer(
            "r",
            torch.tensor(r, dtype=dtype)
        )

        self.register_buffer(
            "sigma",
            torch.tensor(sigma, dtype=dtype)
        )

        # Build the neural network
        layers = []

        # Input layer: two inputs, t and S
        layers.append(
            nn.Linear(2, hidden_width)
        )
        layers.append(nn.Tanh())

        # Additional hidden layers
        for _ in range(hidden_layers - 1):
            layers.append(
                nn.Linear(hidden_width, hidden_width)
            )
            layers.append(nn.Tanh())

        # Output layer: normalized price u(t, S)
        layers.append(
            nn.Linear(hidden_width, 1)
        )

        self.model = nn.Sequential(*layers)

        self.initialize_weights()

    def initialize_weights(self):
        """
        Xavier initialization for all linear layers.
        """
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def normalize_inputs(
        self,
        t: torch.Tensor,
        S: torch.Tensor
    ):
        """
        Normalize t and S to [-1, 1].
        """

        t_scaled = (
            2.0
            * (t - self.t_min_tensor)
            / (self.t_max_tensor - self.t_min_tensor)
            - 1.0
        )

        S_scaled = (
            2.0
            * (S - self.S_min_tensor)
            / (self.S_max_tensor - self.S_min_tensor)
            - 1.0
        )

        return t_scaled, S_scaled

    def forward(
        self,
        t: torch.Tensor,
        S: torch.Tensor
    ):
        """
        Predict normalized option price u(t, S) = V(t, S) / K.
        """

        t_scaled, S_scaled = self.normalize_inputs(
            t,
            S
        )

        inputs = torch.cat(
            (t_scaled, S_scaled),
            dim=1
        )

        u = self.model(inputs)

        return u
