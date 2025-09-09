from .base.dag_experiment_base import DAGSyntheticTestFunction
from typing import Dict, Optional, Tuple, List
import torch


class Schwefel(DAGSyntheticTestFunction):
    """Schwefel function: f(x) = 418.9829d - sum_{i=1}^{d} (x_i * sin(sqrt(|x_i|)))."""

    _optimal_value = 0.0

    def __init__(
        self,
        dim: int,
        observation_noise_std: Optional[float] = None,
        process_stochasticity_std: Optional[float] = None,
        bounds: Optional[List[Tuple[float, float]]] = None,
    ):
        """Initialize the Schwefel function."""
        self.a = 418.9829
        self.optimizers = 420.9687 * torch.ones(dim)

        if bounds is None:
            bounds = [(400, 500) for _ in range(dim)]

        observed_output_node_names: List[str] = ["y1", "y2"]
        root_node_indices_dict: Dict[str, List[int]] = {"x": list(range(dim))}
        objective_node_name = "y2"

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

        # y1: vector of x_i * sin(sqrt(|x_i|)) for each dimension
        y1 = x * torch.sin(torch.sqrt(torch.abs(x)))  # Shape: N x D

        # Add process noise to intermediate output if stochastic
        if self.is_stochastic:
            y1 = y1 + torch.randn_like(y1) * self.process_stochasticity_std

        # y2: final output - sum over all dimensions (using potentially stochastic y1)
        y2 = self.a * self.dim - torch.sum(y1, dim=-1, keepdim=True)  # Shape: N x 1

        # Add process noise to intermediate outputs if stochastic
        if self.is_stochastic:
            y2 = y2 + torch.randn_like(y2) * self.process_stochasticity_std

        return {"y1": y1, "y2": y2}

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

        return output
