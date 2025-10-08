"""Unit tests for utility functions in pogpn.utils."""

import torch
from pogpn.utils import (
    convert_tensor_to_dict,
    convert_dict_to_tensor,
    consolidate_mvn_mixture,
    consolidate_mtmvn_mixture,
    handle_nans_and_create_mask,
)
import gpytorch


def test_convert_tensor_dict_conversion():
    """Test conversion from tensor to dict and back."""
    combined_tensor = torch.randn(10, 5)
    node_indices_dict = {"node1": [0, 2, 4], "node2": [1, 3]}
    tensor_dict = convert_tensor_to_dict(combined_tensor, node_indices_dict)
    assert "node1" in tensor_dict
    assert "node2" in tensor_dict
    assert tensor_dict["node1"].shape == (10, 3)
    assert tensor_dict["node2"].shape == (10, 2)
    reconstructed_tensor = convert_dict_to_tensor(tensor_dict, node_indices_dict)
    assert torch.allclose(combined_tensor, reconstructed_tensor)


def test_consolidate_mvn_mixture():
    """Test consolidation of a mixture of MVNs."""
    m, d = 5, 3
    means = torch.randn(m, d)
    covs = torch.randn(m, d, d)
    covs = covs @ covs.transpose(-1, -2) + torch.eye(d) * 1e-3  # make them PSD
    mvn_batch = gpytorch.distributions.MultivariateNormal(means, covs)
    consolidated_mvn = consolidate_mvn_mixture(mvn_batch)
    assert consolidated_mvn.mean.shape == (d,)
    assert consolidated_mvn.covariance_matrix.shape == (d, d)


def test_consolidate_mtmvn_mixture():
    """Test consolidation of a mixture of multitask MVNs."""
    m, n, t = 5, 10, 2
    mus = torch.randn(m, n, t)
    covs = torch.randn(m, n * t, n * t)
    covs = covs @ covs.transpose(-1, -2) + torch.eye(n * t) * 1e-3
    mtmvn_batch = gpytorch.distributions.MultitaskMultivariateNormal(mus, covs)
    consolidated_mtmvn = consolidate_mtmvn_mixture(mtmvn_batch)
    assert consolidated_mtmvn.mean.shape == (n, t)
    assert consolidated_mtmvn.covariance_matrix.shape == (n * t, n * t)


def test_handle_nans_and_create_mask():
    """Test NaN handling and mask creation."""
    data_dict = {
        "a": torch.tensor([[1.0, 2.0], [3.0, float("nan")]]),
        "b": torch.tensor([[5.0, 6.0], [7.0, 8.0]]),
        "c": torch.tensor([[float("nan"), 1.0], [2.0, 3.0]]),
    }
    imputed_dict, masks_dict = handle_nans_and_create_mask(
        data_dict, imputation_value=-1.0
    )
    assert "a" in masks_dict
    assert "c" in masks_dict
    assert "b" not in masks_dict
    assert torch.all(masks_dict["a"] == torch.tensor([False, True]))
    assert torch.all(masks_dict["c"] == torch.tensor([True, False]))
    assert not torch.isnan(imputed_dict["a"]).any()
    assert not torch.isnan(imputed_dict["c"]).any()
    assert imputed_dict["a"][1, 1] == -1.0
    assert imputed_dict["c"][0, 0] == -1.0
