from .base.dag_experiment_base import DAGSyntheticTestFunction
from typing import Dict, Optional, Tuple, List
import torch


class Griewank(DAGSyntheticTestFunction):
    """Griewank function: f(x) = 1 + sum_{i=1}^{d} (x_i^2 / 4000) - prod_{i=1}^{d} cos(x_i / sqrt(i))."""

    _optimal_value = 0.0

    def __init__(
        self,
        dim: int,
        observation_noise_std: Optional[float] = None,
        process_stochasticity_std: Optional[float] = None,
        bounds: Optional[List[Tuple[float, float]]] = None,
    ):
        """Initialize the Griewank function."""
        self.optimizers = torch.zeros(dim)

        if bounds is None:
            bounds = [(-100, 100) for _ in range(dim)]

        observed_output_node_names: List[str] = ["y1", "y2", "y3"]
        root_node_indices_dict: Dict[str, List[int]] = {"x": list(range(dim))}
        objective_node_name = "y3"

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
        x = input_dict["x"]  # Shape: N x D

        # y1: summation term sum_{i=1}^{d} (x_i^2 / 4000)
        y1 = torch.sum(x**2 / 4000, dim=-1, keepdim=True)  # Shape: N x 1

        # y2: product term prod_{i=1}^{d} cos(x_i / sqrt(i))
        # Create sqrt(i) for each dimension
        sqrt_indices = torch.sqrt(
            torch.arange(1, self.dim + 1, dtype=x.dtype, device=x.device)
        )
        y2 = torch.prod(
            torch.cos(x / sqrt_indices), dim=-1, keepdim=True
        )  # Shape: N x 1

        if self.is_stochastic:
            y1 = y1 + torch.randn_like(y1) * self.process_stochasticity_std
            y2 = y2 + torch.randn_like(y2) * self.process_stochasticity_std

        # y3: final output 1 + y1 - y2 (using potentially stochastic y1, y2)
        y3 = 1 + y1 - y2  # Shape: N x 1

        # Add process noise to intermediate outputs if stochastic
        if self.is_stochastic:
            y3 = y3 + torch.randn_like(y3) * self.process_stochasticity_std

        return {"y1": y1, "y2": y2, "y3": y3}

    def _evaluate_noisy(
        self, input_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Evaluate the noisy function (observation noise only)."""
        output = self._evaluate_true(input_dict)

        # Add observation noise
        if self.observation_noise_std is not None:
            output["y1"] = (
                output["y1"]
                + torch.randn_like(output["y1"]) * self.observation_noise_std
            )
            output["y2"] = (
                output["y2"]
                + torch.randn_like(output["y2"]) * self.observation_noise_std
            )
            output["y3"] = (
                output["y3"]
                + torch.randn_like(output["y3"]) * self.observation_noise_std
            )

        return output
