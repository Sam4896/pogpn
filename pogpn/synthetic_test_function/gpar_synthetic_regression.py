from typing import Optional, Tuple, List, Dict
import torch
from .base.dag_experiment_base import DAGSyntheticTestFunction
import math


class GPARSyntheticRegression(DAGSyntheticTestFunction):
    """GPAR synthetic test problem with a DAG structure.

    This implements the synthetic test problem from the GPAR paper with a DAG structure:

    input_x → y1 → y2 → y3

    Where:
    y1 = -sin(10π(x+1))/(2x+1) - x^4 + stochastic_noise
    y2 = cos(y1)^2 + sin(3x) + stochastic_noise
    y3 = y2 * y1^2 + 3x + stochastic_noise

    Each node can have observation noise added to it.
    """

    def __init__(
        self,
        observation_noise_std: float,
        process_stochasticity_std: float,
        negate: bool = False,
        bounds: Optional[List[Tuple[float, float]]] = None,
        seed: Optional[int] = None,
    ):
        dim = 1
        if bounds is None:
            bounds = [(-1, 1) for _ in range(dim)]
        self._bounds = bounds

        observed_output_node_names = ["y1", "y2", "y3"]
        root_node_dims = {"x": dim}

        objective_node_name = "y3"

        super().__init__(
            dim=dim,
            bounds=bounds,
            observed_output_node_names=observed_output_node_names,
            root_node_dims=root_node_dims,
            objective_node_name=objective_node_name,
            negate=negate,
            process_stochasticity_std=process_stochasticity_std,
            observation_noise_std=observation_noise_std,
            seed=seed,
        )

    def _evaluate_true(
        self, input_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        x = input_dict["x"].to(torch.float64)

        # Ensure X is the right shape
        if x.ndim == 1:
            x = x.unsqueeze(-1)

        # Calculate y1
        y1 = -torch.sin(10 * math.pi * (x + 1)) / (2 * x + 1) - x**4
        if self.is_stochastic:
            y1 = y1 + x * self.process_stochasticity_std * torch.randn_like(x)

        # Calculate y2
        y2 = torch.cos(y1) ** 2 + torch.sin(3 * x)
        if self.is_stochastic:
            y2 = y2 + x * self.process_stochasticity_std * torch.randn_like(x)

        # Calculate y3
        y3 = y2 * y1**2 + 3 * x
        if self.is_stochastic:
            y3 = y3 + x * self.process_stochasticity_std * torch.randn_like(x)

        # Return as dictionary
        return {
            "y1": y1,
            "y2": y2,
            "y3": y3,
        }

    def _evaluate_noisy(
        self, input_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        output = self._evaluate_true(input_dict)
        for key in output:
            output[key] = (
                output[key] + torch.randn_like(output[key]) * self.observation_noise_std
            )
        return output
