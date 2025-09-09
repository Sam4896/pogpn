from .gpar_synthetic_regression import GPARSyntheticRegression
import torch
from typing import Dict


class GPARSyntheticClassification(GPARSyntheticRegression):
    """GPAR synthetic test problem with classification and a DAG structure.

    This implements a synthetic test problem with a DAG structure:

    input_x → y1 → y2 → y3

    Where:
    y1 = -sin(10π(x+1))/(2x+1) - x^4 + stochastic_noise
    y2 = 1 if cos(y1)^2 + sin(3x) + stochastic_noise >= 1.5 else 0
    y3 = y2 * y1^2 + 3x + stochastic_noise.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def evaluate_noisy(
        self, input_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Evaluate the noisy function values at X with stochastic noise and observation likelihood.

        y2 is a classification node, so we need to convert it to a binary classification problem.
        """
        x = input_dict["x"].to(torch.float64)

        # Ensure X is the right shape
        if x.ndim == 1:
            x = x.unsqueeze(-1)

        # Get the stochastic data dictionary
        data_dict = self._evaluate_true(input_dict)

        # Convert y2 to binary classification (0 or 1) based on threshold of 1.5
        # Get the continuous value of y2 before thresholding
        y2_continuous = data_dict["y2"].squeeze(-1)

        # Apply threshold to create binary classification
        cat_0_idx_y = y2_continuous < 1.5
        cat_1_idx_y = y2_continuous >= 1.5

        # Create binary classification output
        y2_binary = torch.zeros_like(y2_continuous)
        y2_binary[cat_0_idx_y] = 0.0
        y2_binary[cat_1_idx_y] = 1.0

        # Update y2 in the data dictionary
        data_dict["y2"] = y2_binary  # Don't unsqueeze classification vectors.

        # Add observation noise to y1 and y3 (not to y2 since it's classification)
        if self.observation_noise_std is not None:
            data_dict["y1"] = data_dict[
                "y1"
            ] + x * self.observation_noise_std * torch.randn_like(x)
            # No observation noise for y2 as it's a classification node
            data_dict["y3"] = data_dict[
                "y3"
            ] + x * self.observation_noise_std * torch.randn_like(x)

        # Return the noisy data dictionary
        return data_dict
