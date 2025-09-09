from typing import Dict, List

import torch
from botorch.posteriors import GPyTorchPosterior
import gpytorch


def convert_tensor_to_dict(
    combined_tensor: torch.Tensor,
    node_indices_dict: Dict[str, List[int]],
    dim: int = -1,
) -> Dict[str, torch.Tensor]:
    """Convert a tensor to a dictionary of tensors based on index lists.

    Args:
        combined_tensor: Tensor of shape (n, d) where n is the number of samples and d is the total dimension of the input nodes.
        node_indices_dict: Dictionary mapping node names to the list of indices in combined_tensor that belong to that node.
        dim: Dimension to index along (should be -1 for last dim).

    Returns:
        Dictionary of tensors with the node names as keys and the corresponding tensor slices as values.

    """
    out = {}
    for node, idx in node_indices_dict.items():
        out[node] = combined_tensor.index_select(
            dim, torch.tensor(idx, device=combined_tensor.device)
        )
    return out


def convert_dict_to_tensor(
    node_data_dict: Dict[str, torch.Tensor],
    node_indices_dict: Dict[str, List[int]],
) -> torch.Tensor:
    """Place each tensor's channels into the combined tensor at the given indices.

    along the last dimension. Works with shapes (..., C_k) sharing the same
    leading dimensions.

    Example: node_data_dict = {'x1': (N,3), 'x2': (N,2)}
             node_indices_dict = {'x1': [0,1,4], 'x2': [2,3]}
             -> output shape (N, 5), ordered [x1_0, x1_1, x2_0, x2_1, x1_2]
    """
    # Use the first tensor to infer leading dims, dtype, device
    first = next(iter(node_data_dict.values()))
    leading = first.shape[:-1]
    total_C = sum(t.shape[-1] for t in node_data_dict.values())

    out = first.new_zeros(*leading, total_C)

    for k, t in node_data_dict.items():
        if t.shape[:-1] != leading:
            raise ValueError(
                f"All tensors must share leading shape {leading}, got {t.shape}."
            )
        idx = torch.as_tensor(node_indices_dict[k], dtype=torch.long, device=out.device)
        if t.shape[-1] != idx.numel():
            raise ValueError(
                f"Mismatched channels for '{k}': got {t.shape[-1]} vs {idx.numel()} indices."
            )
        # Copy channels into their slots along the last dimension
        out.index_copy_(-1, idx, t)

    return out


def consolidate_mvn_mixture(mvn_batch, weights=None, eps=1e-6):
    """Reduce a batch of MVNs (batch dim M) to a single MVN via moment matching.

    dist.mean: shape [M, D]
    dist.covariance_matrix: shape [M, D, D]
    weights: optional [M] (unnormalized OK). If None, uses uniform.
    """
    if isinstance(mvn_batch, GPyTorchPosterior):
        is_posterior = True
        dist = mvn_batch.distribution
    else:
        is_posterior = False
        dist = mvn_batch

    mus = dist.mean  # [M, D]
    covs = dist.covariance_matrix  # [M, D, D]
    M, D = mus.shape

    if weights is None:
        weights = torch.full((M,), 1.0 / M, dtype=mus.dtype, device=mus.device)
    else:
        weights = weights.to(mus) / weights.sum()

    w = weights.view(M, 1)
    mu = (w * mus).sum(dim=0)  # [D]

    # Σ = Σ_i w_i (Σ_i + μ_i μ_i^T) - μ μ^T
    outer = mus.unsqueeze(-1) @ mus.unsqueeze(-2)  # [M, D, D]
    second = (weights.view(M, 1, 1) * (covs + outer)).sum(dim=0)  # [D, D]
    Sigma = second - (mu.unsqueeze(-1) @ mu.unsqueeze(-2))  # [D, D]

    # Stabilize & symmetrize
    Sigma = 0.5 * (Sigma + Sigma.transpose(-1, -2))
    Sigma = Sigma + eps * torch.eye(D, dtype=Sigma.dtype, device=Sigma.device)

    if is_posterior:
        return GPyTorchPosterior(gpytorch.distributions.MultivariateNormal(mu, Sigma))
    else:
        return gpytorch.distributions.MultivariateNormal(mu, Sigma)


def consolidate_mtmvn_mixture(mtmvn_batch, weights=None, eps: float = 1e-6):
    """Reduce a batch of MTMVNs across the leading batch dim via moment matching.

    Preserves cross-task correlations and the original interleaving layout.

    Args:
        mtmvn_batch: A batch of MTMVNs.
        weights: Optional weights for the MTMVNs.
        eps: Epsilon for numerical stability.

    Returns:
        A consolidated MTMVN (or a GPyTorchPosterior over it if that was the input).

    """
    try:
        from botorch.posteriors.gpytorch import GPyTorchPosterior

        is_posterior = isinstance(mtmvn_batch, GPyTorchPosterior)
    except Exception:
        GPyTorchPosterior = None
        is_posterior = False

    dist = mtmvn_batch.distribution if is_posterior else mtmvn_batch
    assert isinstance(dist, gpytorch.distributions.MultitaskMultivariateNormal)

    interleaved = bool(
        getattr(dist, "interleaved", getattr(dist, "_interleaved", True))
    )
    T = dist.num_tasks

    # mean: [M, N, T] (or [N, T] -> add M=1)
    mus = dist.mean
    if mus.dim() == 2:
        mus = mus.unsqueeze(0)
    M, N, TT = mus.shape
    assert TT == T

    # cov: [M, N*T, N*T] (or [N*T, N*T] -> add M=1)
    covs = dist.covariance_matrix
    if covs.dim() == 2:  # type: ignore
        covs = covs.unsqueeze(0)  # type: ignore
    assert covs.shape == (M, N * T, N * T)  # type: ignore

    # flatten means to match covariance ordering
    mus_flat = (
        mus.reshape(M, N * T) if interleaved else mus.permute(0, 2, 1).reshape(M, N * T)
    )

    # weights over batch components
    if weights is None:
        weights = torch.full(
            (M,), 1.0 / M, dtype=mus_flat.dtype, device=mus_flat.device
        )
    else:
        weights = weights.to(mus_flat).reshape(M)
        weights = weights / weights.sum()

    # moment matching in NT-dimensional space
    w = weights.view(M, 1)
    mu_flat = (w * mus_flat).sum(dim=0)  # [NT]
    outer = mus_flat.unsqueeze(-1) @ mus_flat.unsqueeze(-2)  # [M, NT, NT]
    second = (weights.view(M, 1, 1) * (covs + outer)).sum(dim=0)  # [NT, NT]
    Sigma = second - (mu_flat.unsqueeze(-1) @ mu_flat.unsqueeze(-2))  # [NT, NT]

    # stabilize
    Sigma = 0.5 * (Sigma + Sigma.transpose(-1, -2))
    Sigma = Sigma + eps * torch.eye(N * T, dtype=Sigma.dtype, device=Sigma.device)

    # reshape mean to [N, T] consistent with layout; covariance stays [NT, NT]
    if interleaved:
        mu_2d = mu_flat.view(N, T)
    else:
        mu_2d = mu_flat.view(T, N).transpose(-1, -2)  # -> [N, T]

    consolidated = gpytorch.distributions.MultitaskMultivariateNormal(
        mu_2d, Sigma, interleaved=interleaved
    )

    return GPyTorchPosterior(consolidated) if is_posterior else consolidated  # type: ignore


def handle_nans_and_create_mask(data_dict, imputation_value=-999.0):
    """Create a mask for NaNs in the data_dict and impute them.

    Args:
        data_dict: A dictionary of tensors.
        imputation_value: The value to use for imputing NaNs.

    Returns:
        A tuple containing:
            - imputed_data_dict: The data dictionary with NaNs imputed.
            - masks_dict: A dictionary of boolean masks, where True indicates a row
                that originally contained a NaN.

    """
    imputed_data_dict = {}
    masks_dict = {}
    for key, tensor in data_dict.items():
        if isinstance(tensor, torch.Tensor):
            imputed_tensor = tensor.clone()
            if torch.isnan(imputed_tensor).any():
                row_mask = torch.isnan(imputed_tensor).any(dim=-1)
                masks_dict[key] = row_mask
                imputed_tensor[row_mask] = imputation_value
                imputed_data_dict[key] = imputed_tensor
            else:
                imputed_data_dict[key] = imputed_tensor
        else:
            imputed_data_dict[key] = tensor

    return imputed_data_dict, masks_dict
