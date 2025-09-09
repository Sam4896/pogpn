from .base.dag_experiment_base import (
    DAGSyntheticTestFunction,
)
from typing import Dict, Optional, Tuple, List
import torch
import math


class Levy(DAGSyntheticTestFunction):
    """Levy function.

    f(x) = sin²(πw₁) + Σ_{i=1}^{d-1} ((wᵢ-1)² [1 + 10 sin²(πwᵢ + 1)]) + (w_d-1)² [1 + sin²(2πw_d)],
    where wᵢ = 1 + (xᵢ - 1) / 4.
    """

    _optimal_value = 0.0

    def __init__(
        self,
        dim: int,
        observation_noise_std: Optional[float] = None,
        process_stochasticity_std: Optional[float] = None,
        bounds: Optional[List[Tuple[float, float]]] = None,
    ):
        """Initialize the Levy function."""
        if bounds is None:
            bounds = [(-10.0, 10.0) for _ in range(dim)]

        observed_output_node_names: List[str] = ["y1", "y2", "y3", "y4"]
        root_node_indices_dict: Dict[str, List[int]] = {
            "x1": [0],
            "x2": list(range(1, dim - 1)),
            "x3": [dim - 1],
        }
        objective_node_name = "y4"

        self.optimizers = torch.ones(dim)

        super().__init__(
            dim=dim,
            bounds=bounds,
            observed_output_node_names=observed_output_node_names,
            root_node_indices_dict=root_node_indices_dict,
            objective_node_name=objective_node_name,
            negate=True,
            process_stochasticity_std=process_stochasticity_std,
            observation_noise_std=observation_noise_std,
        )

    def _evaluate_true(
        self, input_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Evaluate the true function (no noise)."""
        # Extract inputs: x1=x_1, x2=(x_2,...,x_{d-1}), x3=x_d
        x1 = input_dict["x1"]  # Shape: N x 1 (first dimension)
        x2 = input_dict["x2"]  # Shape: N x (D-2) (middle dimensions, if any)
        x3 = input_dict["x3"]  # Shape: N x 1 (last dimension)

        # Transform inputs to w: wᵢ = 1 + (xᵢ - 1) / 4
        w1 = 1 + (x1 - 1) / 4  # Shape: N x 1
        w2 = 1 + (x2 - 1) / 4  # Shape: N x (D-2)
        w3 = 1 + (x3 - 1) / 4  # Shape: N x 1

        # y1: term associated with w₁ (first dimension)
        y1 = torch.sin(math.pi * w1) ** 2  # Shape: N x 1

        # y2: summation term for w₁ to w_{d-1}
        sum_term_w1 = (w1 - 1) ** 2 * (1 + 10 * torch.sin(math.pi * w1 + 1) ** 2)
        sum_term_w2 = torch.sum(
            (w2 - 1) ** 2 * (1 + 10 * torch.sin(math.pi * w2 + 1) ** 2),
            dim=-1,
            keepdim=True,
        )  # Shape: N x 1
        y2 = sum_term_w1 + sum_term_w2

        # y3: term associated with w_d (last dimension)
        y3 = (w3 - 1) ** 2 * (1 + torch.sin(2 * math.pi * w3) ** 2)  # Shape: N x 1

        # Add process noise to intermediate outputs if stochastic
        if self.is_stochastic:
            y1 = y1 + torch.randn_like(y1) * self.process_stochasticity_std
            y2 = y2 + torch.randn_like(y2) * self.process_stochasticity_std
            y3 = y3 + torch.randn_like(y3) * self.process_stochasticity_std

        # y4: final output - sum of all terms (using potentially stochastic y1, y2, y3)
        y4 = y1 + y2 + y3  # Shape: N x 1

        # Add process noise to final output if stochastic
        if self.is_stochastic:
            y4 = y4 + torch.randn_like(y4) * self.process_stochasticity_std

        return {"y1": y1, "y2": y2, "y3": y3, "y4": y4}

    def _evaluate_noisy(
        self, input_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Evaluate the noisy function (observation noise only)."""
        output = self._evaluate_true(input_dict)

        # Add observation noise
        if self.observation_noise_std is not None:
            for key in ["y1", "y2", "y3", "y4"]:
                output[key] = (
                    output[key]
                    + torch.randn_like(output[key]) * self.observation_noise_std
                )

        return output
