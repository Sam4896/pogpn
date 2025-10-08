"""Unit tests for synthetic test functions."""

import pytest
import torch

from pogpn.synthetic_test_function import (
    Ackley,
    CatalyticBatchReactor,
    Griewank,
    Levy,
    Michalewicz,
    PenicillinJPSS,
    Rosenbrock,
    Schwefel,
    ServiceNetworkPCDirect,
    ServiceNetworkPCDirectEnterprise,
)

# Functions that are expected to instantiate and run correctly
WORKING_FUNCTIONS = [
    (Ackley, {"dim": 4}),
    (CatalyticBatchReactor, {}),
    (Griewank, {"dim": 4}),
    (Levy, {"dim": 4}),
    (Michalewicz, {"dim": 5}),
    (PenicillinJPSS, {}),
    (Rosenbrock, {"dim": 4}),
    (Schwefel, {"dim": 4}),
    (ServiceNetworkPCDirect, {}),
    (ServiceNetworkPCDirectEnterprise, {}),
]


@pytest.mark.parametrize("func_class, kwargs", WORKING_FUNCTIONS)
def test_synthetic_function_smoke_test(func_class, kwargs):
    """A simple smoke test to ensure synthetic functions can be instantiated and evaluated."""
    # Instantiate the function
    func = func_class(**kwargs)

    # Create a sample input tensor
    X = func.get_sobol_samples(3)
    assert X.shape == (3, func.dim)

    # Evaluate the function
    output = func(X)

    # Perform basic checks on the output
    assert isinstance(output, dict)
    for key in func.observed_output_node_names:
        assert key in output
        assert isinstance(output[key], torch.Tensor)
        assert output[key].shape[0] == 3


@pytest.mark.parametrize("func_class, kwargs", WORKING_FUNCTIONS)
def test_synthetic_function_noise_and_determinism(func_class, kwargs):
    """Tests the noise application and determinism of the synthetic functions."""
    # Instantiate a clean version of the function (no noise)
    kwargs_clean = kwargs.copy()
    kwargs_clean.update(
        {"observation_noise_std": 0.0, "process_stochasticity_std": 0.0}
    )
    func_clean = func_class(**kwargs_clean)

    # Instantiate a noisy version of the function
    kwargs_noisy = kwargs.copy()
    kwargs_noisy.update(
        {"observation_noise_std": 0.1, "process_stochasticity_std": 0.1}
    )
    func_noisy = func_class(**kwargs_noisy)

    # Create a sample input tensor
    X = func_clean.get_sobol_samples(1)

    # Test that the clean function is deterministic
    output1_clean = func_clean(X)
    output2_clean = func_clean(X)
    torch.testing.assert_close(
        output1_clean[func_clean.objective_node_name],
        output2_clean[func_clean.objective_node_name],
    )

    # Test that noise is applied and the output is different from the clean version
    output_noisy = func_noisy(X)
    assert not torch.allclose(
        output1_clean[func_clean.objective_node_name],
        output_noisy[func_noisy.objective_node_name],
    )

    # Test that the noisy function is stochastic (produces different outputs on each call)
    output1_noisy = func_noisy(X)
    output2_noisy = func_noisy(X)
    assert not torch.allclose(
        output1_noisy[func_noisy.objective_node_name],
        output2_noisy[func_noisy.objective_node_name],
    )
