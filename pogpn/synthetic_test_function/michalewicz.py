from .base.dag_experiment_base import (
    DAGSyntheticTestFunction,
)
from typing import Dict, Optional, Tuple, List
import math
import torch


class Michalewicz(DAGSyntheticTestFunction):
    """Michalewicz function."""

    def __init__(
        self,
        dim: int,
        observation_noise_std: Optional[float] = None,
        process_stochasticity_std: Optional[float] = None,
        bounds: Optional[List[Tuple[float, float]]] = None,
    ):
        """Initialize the Michalewicz function."""
        self.m = 10.0

        if bounds is None:
            bounds = [(0.0, math.pi) for _ in range(dim)]

        observed_output_node_names: List[str] = ["y1", "y2", "y3"]
        root_node_indices_dict: Dict[str, List[int]] = {"x": list(range(dim))}
        objective_node_name = "y3"

        if dim == 2:
            self.optimizers = torch.tensor([2.20, 1.57])
            self._optimal_value = -1.8013
        elif dim == 5:
            self.optimizers = torch.tensor([2.20, 1.57, 2.80, 1.71, 2.30])
            self._optimal_value = -4.687658
        elif dim == 10:
            self.optimizers = torch.tensor(
                [2.20, 1.57, 2.80, 1.71, 2.30, 2.60, 2.30, 2.90, 2.40, 2.50]
            )
            self._optimal_value = -9.6601515
        else:
            self.optimizers = None
            self._optimal_value = None

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
        x = input_dict["x"]
        dim_arr = torch.arange(1, self.dim + 1, 1, device=x.device, dtype=x.dtype)

        y1_base = torch.sin(x)

        if self.is_stochastic:
            part1 = self._add_proportional_noise(
                y1_base, self.process_stochasticity_std
            )
        else:
            part1 = y1_base

        y2_base = torch.sin(dim_arr * x.pow(2) / torch.pi).pow(2 * self.m)
        if self.is_stochastic:
            part2 = self._add_proportional_noise(
                y2_base, self.process_stochasticity_std
            )
        else:
            part2 = y2_base

        y3 = -torch.sum(part1 * part2, dim=-1)
        return {"y1": part1, "y2": part2, "y3": y3}

    def _evaluate_noisy(
        self, input_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Evaluate the noisy function (process noise + observation noise)."""
        output = self._evaluate_true(input_dict)
        for key in ["y1", "y2", "y3"]:
            output[key] = self._add_proportional_noise(
                output[key], self.observation_noise_std
            )
        return output
