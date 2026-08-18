import torch
import torch.nn as nn
import torch.nn.functional as F


# Use the same dtype as Notebook 02
dtype = torch.float64


def inverse_softplus(x):

    x = torch.as_tensor(
        x,
        dtype=dtype
    )

    return torch.log(
        torch.expm1(x)
    )


class InversePINN(nn.Module):
    """
    Inverse PINN for the European call Black-Scholes problem.

    Inputs:
        t = current time
        S = current spot price

    Output:
        u(t, S) = V(t, S) / K

    Unknown trainable PDE parameters:
        r
        sigma
    """

    def __init__(
        self,
        t_min: float,
        t_max: float,
        S_min: float,
        S_max: float,
        hidden_width: int = 32,
        hidden_layers: int = 4,
        r_initial: float = 0.10,
        sigma_initial: float = 0.40
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

        # Unknown trainable PDE parameters
        self.r_raw = nn.Parameter(
            inverse_softplus(r_initial)
        )

        self.sigma_raw = nn.Parameter(
            inverse_softplus(sigma_initial)
        )

        self.initialize_weights()

    def initialize_weights(self):
        """
        Xavier initialization for all linear layers.
        """
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    @property
    def r(self):
        """
        Positive estimate of the risk-free interest rate.
        """
        return F.softplus(self.r_raw)

    @property
    def sigma(self):
        """
        Positive estimate of volatility.
        """
        return F.softplus(self.sigma_raw)

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
        Predict normalized option price u(t, S).
        """

        t_scaled, S_scaled = self.normalize_inputs(t, S)

        inputs = torch.cat(
            (t_scaled, S_scaled),
            dim=1
        )

        u = self.model(inputs)

        return u