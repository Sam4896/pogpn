from abc import abstractmethod, ABC
from typing import Dict, List, Optional, Tuple

import torch
from botorch.utils.transforms import unnormalize

from ...pogpn.utils import convert_tensor_to_dict
from botorch.test_functions.synthetic import SyntheticTestFunction


class DAGSyntheticTestFunction(SyntheticTestFunction, ABC):
    """Base class for experiment functions.

    This abstract base class provides the foundation for all experiment functions.
    It handles input and output validation while requiring subclasses to implement
    the actual evaluation logic.

    Args:
        dim: The dimension of the input space.
        negate: Whether to negate the function.
        bounds: The bounds of the input space.
        device: The device to run the function on.
        is_observation_noisy: Whether to add observation noise.
        is_stochastic: Whether to add process noise.
        observed_output_node_names: The names of the observed output nodes.
        root_node_indices: The indices of the root nodes.

    """

    def __init__(
        self,
        dim: int,
        negate: bool,
        bounds: List[Tuple[float, float]],
        observed_output_node_names: List[str],
        root_node_indices_dict: Dict[str, List[int]],
        objective_node_name: str,
        process_stochasticity_std: Optional[float] = None,
        observation_noise_std: Optional[float] = None,
    ):
        self.dim = dim
        self.continuous_inds = list(range(dim))

        super().__init__(negate=negate, bounds=bounds)

        self.observed_output_node_names = observed_output_node_names
        self.root_node_indices_dict = root_node_indices_dict
        self.objective_node_name = objective_node_name

        # Set process stochasticity
        self.process_stochasticity_std = (
            process_stochasticity_std if process_stochasticity_std is not None else 0.0
        )
        self.is_stochastic = (
            process_stochasticity_std is not None and process_stochasticity_std > 0.0
        )

        # Set observation noise
        self.observation_noise_std = (
            observation_noise_std if observation_noise_std is not None else 0.0
        )
        self.is_observation_noisy = (
            observation_noise_std is not None and observation_noise_std > 0.0
        )

        # Find optimum
        if not hasattr(self, "_optimal_value"):
            self.optimizers, self._optimal_value = (
                self.find_optimum_multi_start_high_dim()
            )

        self._validate_root_node_indices()

    def _validate_root_node_indices(self):
        if self.root_node_indices_dict is None:
            raise ValueError("root_node_indices_dict must be set.")
        # root_node_indices is Dict[str, List[int]], all indices must be unique and cover range(dim)
        all_inds = [i for inds in self.root_node_indices_dict.values() for i in inds]
        if sorted(all_inds) != list(range(self.dim)):
            raise ValueError(
                f"root_node_indices must cover all input indices 0..{self.dim - 1} exactly once. Got: {all_inds}"
            )

    def _validate_output_shape(self, output: torch.Tensor) -> torch.Tensor:
        """Validate the output shape of the function."""
        if output.ndim == 1:
            output = output.unsqueeze(-1)
        return output

    @abstractmethod
    def _evaluate_true(
        self, input_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Evaluate the function without noise."""
        pass

    @abstractmethod
    def _evaluate_noisy(
        self, input_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Evaluate the function with system and observation noise."""
        pass

    def forward(self, X: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Evaluate the function on a batch of points X.

        Args:
            X: A `batch_shape x d`-dim tensor of inputs.

        Returns:
            A dictionary of function evaluations.

        """
        if X.shape[-1] != self.dim:
            raise ValueError(f"Input dimension must be {self.dim}")

        input_dict = convert_tensor_to_dict(
            combined_tensor=X, node_indices_dict=self.root_node_indices_dict
        )

        if self.is_observation_noisy:
            output = self._evaluate_noisy(input_dict)
        else:
            output = self._evaluate_true(input_dict)

        for key in self.observed_output_node_names:
            if self.negate:
                output[key] = -self._validate_output_shape(output[key])
            else:
                output[key] = self._validate_output_shape(output[key])

        output["inputs"] = X

        return output

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"  dim={self.dim},\n"
            f"  negate={self.negate},\n"
            f"  device={self.device},\n"
            f"  is_observation_noisy={self.is_observation_noisy},\n"
            f"  is_stochastic={self.is_stochastic},\n"
            f"  observation_noise_std={self.observation_noise_std},\n"
            f"  process_stochasticity_std={self.process_stochasticity_std},\n"
            f"  bounds={self.bounds},\n"
            f"  observed_output_node_names={self.observed_output_node_names},\n"
            f"  root_node_indices_dict={self.root_node_indices_dict},\n"
            f")"
        )

    def get_output_tensor(
        self,
        outputs_dict: Dict[str, torch.Tensor],
        node_order: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """Convert outputs from dictionary to concatenated tensor.

        Args:
            outputs_dict: Dictionary containing output tensors
            node_order: Optional list of node names to order the output tensor by

        Returns:
            A tensor of all outputs concatenated along the last dimension

        """
        output_tensors = []
        if node_order is None:
            node_order = sorted(self.observed_output_node_names)
        for node_name in node_order:
            if node_name not in outputs_dict:
                raise ValueError(f"Missing output for node {node_name}")
            output_tensors.append(outputs_dict[node_name])

        return torch.cat(output_tensors, dim=-1)

    def get_sobol_samples(self, n_samples: int) -> torch.Tensor:
        """Get Sobol samples from the bounds of the sim env.

        Args:
            n_samples: The number of samples to generate.

        Returns:
            A `n_samples x dim` tensor of Sobol samples.
        """
        device, dtype = self.bounds.device, self.bounds.dtype
        sobol = torch.quasirandom.SobolEngine(dimension=self.dim, scramble=True)
        samples = sobol.draw(n_samples).to(device=device, dtype=dtype)  # type: ignore
        samples = unnormalize(samples, self.bounds)  # type: ignore
        return samples

    def find_optimum_multi_start_high_dim(self, n_starts=10, n_steps=500):
        """Find the optimum of the function using multi-start optimization."""
        dim = self.dim  # 100 in this case
        bounds = self.bounds
        device, dtype = bounds.device, bounds.dtype

        best_value = float("-inf")
        best_point = None

        # Generate quasi-random starting points instead of purely random
        sobol = torch.quasirandom.SobolEngine(dimension=dim, scramble=True)
        start_points = sobol.draw(n_starts).to(device=device, dtype=dtype)  # type: ignore
        # start_points = bounds[0] + (bounds[1] - bounds[0]) * start_points
        start_points = unnormalize(start_points, bounds)  # type: ignore

        for i in range(n_starts):
            x = start_points[i, :].unsqueeze(-2)
            x.requires_grad_(True)

            # Use a more aggressive optimization scheme
            optimizer = torch.optim.Adam([x], lr=0.05)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=50, factor=0.5
            )

            prev_loss = float("inf")
            for step in range(n_steps):
                optimizer.zero_grad()
                outputs_dict = self(x)
                obj = -outputs_dict[self.objective_node_name]  # Negate for maximization
                obj.backward()

                # Track progress for scheduler
                if step % 10 == 0:
                    scheduler.step(obj.item())

                    # Early stopping if progress is minimal
                    if abs(prev_loss - obj.item()) < 1e-7:
                        break
                    prev_loss = obj.item()

                optimizer.step()

                # Project back to bounds
                with torch.no_grad():
                    x.data = torch.max(torch.min(x.data, bounds[1]), bounds[0])  # type: ignore

            outputs_dict = self(x)
            value = outputs_dict[self.objective_node_name].item()
            if value > best_value:
                best_value = value
                best_point = x.detach().clone()

        if best_point is None:
            raise ValueError("Optimization failed to find a valid point")
        return best_point.detach().cpu().numpy(), best_value
