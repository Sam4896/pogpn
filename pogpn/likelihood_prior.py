import math
from typing import Optional
import torch
from botorch.models.transforms import Standardize
from botorch.models.transforms.outcome import OutcomeTransform
from gpytorch.priors.torch_priors import LogNormalPrior

SQRT_2PI = math.sqrt(2.0 * math.pi)
DEFAULT_TARGET_DENSITY_AT_MEAN = 20.0


def _scale_for_density_at_mean(
    loc: float, target_density: float, max_iter: int = 25
) -> float:
    """For LogNormal(mu, sigma), pdf at its mean m = exp(mu + 0.5*sigma^2) is.

        pdf(m) = exp(-mu - 5/8*sigma^2) / (sigma*sqrt(2*pi))
    Solve for sigma > 0 given target_density and loc=mu.

    We solve: (5/8)*sigma^2 + log(sigma) = -log(target_density*sqrt(2*pi)) - mu
    via a few Newton iterations, then clamp to a sensible range.
    """
    if target_density <= 0:
        raise ValueError("target_density must be positive.")
    k = -math.log(target_density * SQRT_2PI) - loc

    # tiny, robust Newton with a simple positive initial guess
    sigma = math.exp(k)
    sigma = max(1e-6, min(10.0, sigma))

    for _ in range(max_iter):
        h = 0.625 * sigma * sigma + math.log(sigma) - k
        hp = 1.25 * sigma + 1.0 / sigma
        step = h / hp
        sigma = max(1e-6, min(10.0, sigma - step))
    return float(sigma)


def _rescale_noise_by_standardize_stds(
    noise_std: float, standardize: Standardize
) -> float:
    """Rescale a scalar noise std into standardized space by dividing by stds.

    Works for batch-shaped stds with shape (*B, M). Returns a scalar by averaging
    across batch/output dims, which mirrors the previous scalar reduction.
    """
    stds = standardize.stds  # shape: (*B, M)
    stds = stds.clamp_min(1e-12)
    scaled = noise_std * torch.ones_like(stds) / stds
    return float(scaled.mean().item())


def get_lognormal_likelihood_prior(
    node_observation_noise: Optional[float] = None,
    target_density_at_mean: float = DEFAULT_TARGET_DENSITY_AT_MEAN,
    node_transform: Optional[OutcomeTransform] = None,
) -> LogNormalPrior:
    """Build a LogNormalPrior whose pdf at its (log-normal) mean equals target_density_at_mean.

    The (intended) mean is the observation-noise std in model space. If a
    Standardize node_transform is provided, we divide the provided noise std by
    the transform stds to map it into the standardized space.
    """
    if node_observation_noise is None or node_observation_noise < 1e-6:
        return LogNormalPrior(loc=-4.0, scale=1.0)

    # start with the user-provided noise std (positive)
    m = float(node_observation_noise)

    # If a Standardize transform is present, rescale noise by its stds only.
    if isinstance(node_transform, Standardize):
        try:
            m = _rescale_noise_by_standardize_stds(m, node_transform)
        except Exception:
            # If anything goes wrong, keep original m
            pass

    m = max(m, 1e-12)

    # Set loc = log(m), then choose scale s.t. pdf at the (log-normal) mean hits the target.
    loc = math.log(m)
    scale = _scale_for_density_at_mean(loc=loc, target_density=target_density_at_mean)
    return LogNormalPrior(loc=loc, scale=scale)
