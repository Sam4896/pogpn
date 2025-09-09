from .base.dag_experiment_base import DAGSyntheticTestFunction
from typing import Dict, Optional, Tuple, List
import math
import torch


class Ackley(DAGSyntheticTestFunction):
    """Ackley function."""

    _optimal_value = 0.0

    def __init__(
        self,
        dim: int,
        observation_noise_std: Optional[float] = None,
        process_stochasticity_std: Optional[float] = None,
        bounds: Optional[List[Tuple[float, float]]] = None,
    ):
        """Initialize the Ackley function."""
        self.a = 20.0
        self.b = 0.2
        self.c = 2 * math.pi

        if bounds is None:
            bounds = [(-32.768, 32.768) for _ in range(dim)]

        observed_output_node_names: List[str] = ["y1", "y2", "y3"]
        root_node_indices_dict: Dict[str, List[int]] = {"x": list(range(dim))}
        objective_node_name = "y3"

        self.optimizers = torch.zeros(dim)

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
        a, b, c = self.a, self.b, self.c
        x = input_dict["x"]
        y1 = -b / math.sqrt(self.dim) * torch.linalg.norm(x, dim=-1)

        if self.is_stochastic:
            part1 = (
                -a * torch.exp(y1)
                + torch.randn_like(y1) * self.process_stochasticity_std
            )
        else:
            part1 = -a * torch.exp(y1)

        y2 = torch.mean(torch.cos(c * x), dim=-1)
        if self.is_stochastic:
            part2 = (
                -torch.exp(y2) + torch.randn_like(y2) * self.process_stochasticity_std
            )
        else:
            part2 = -torch.exp(y2)

        y3 = part1 + part2 + a + math.e
        return {"y1": part1, "y2": part2, "y3": y3}

    def _evaluate_noisy(
        self, input_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Evaluate the noisy function (process noise + observation noise)."""
        output = self._evaluate_true(input_dict)
        for key in ["y1", "y2", "y3"]:
            output[key] = (
                output[key] + torch.randn_like(output[key]) * self.observation_noise_std
            )
        return output
