import torch
from pogpn import POGPNNodewise, POGPNPathwise
import gpytorch


def test_pogpn_nodewise_initialization(simple_dag_and_data):
    """Test the initialization of the POGPNNodewise model."""
    model = POGPNNodewise(
        dag=simple_dag_and_data["dag"],
        data_dict=simple_dag_and_data["data_dict"],
        root_node_indices_dict=simple_dag_and_data["root_node_indices_dict"],
        objective_node_name=simple_dag_and_data["objective_node_name"],
    )
    assert model is not None
    assert "y1" in model.non_root_nodes
    assert "x1" in model.root_nodes
    assert isinstance(
        model.model.node_mlls_dict["y1"], gpytorch.mlls.MarginalLogLikelihood
    )


def test_pogpn_nodewise_fit_and_posterior(simple_dag_and_data):
    """Test fitting the POGPNNodewise model and getting a posterior."""
    with gpytorch.settings.num_likelihood_samples(8):
        model = POGPNNodewise(
            dag=simple_dag_and_data["dag"],
            data_dict=simple_dag_and_data["data_dict"],
            root_node_indices_dict=simple_dag_and_data["root_node_indices_dict"],
            objective_node_name=simple_dag_and_data["objective_node_name"],
        )
        # Test fit
        model.fit(optimizer="torch", maxiter=2, lr=0.01)

        # Test posterior
        test_x = torch.randn(5, 2, dtype=torch.float64)
        posterior = model.posterior(test_x)

        # Test rsample
        samples = posterior.rsample(torch.Size([32]))
        assert samples.shape == (32, 5, 1)

        # Test mean and variance from samples
        mean = samples.mean(dim=0)
        variance = samples.var(dim=0)
        assert mean.shape[-2:] == (5, 1)
        assert variance.shape[-2:] == (5, 1)


def test_pogpn_pathwise_initialization(simple_dag_and_data):
    """Test the initialization of the POGPNPathwise model."""
    model = POGPNPathwise(
        dag=simple_dag_and_data["dag"],
        data_dict=simple_dag_and_data["data_dict"],
        root_node_indices_dict=simple_dag_and_data["root_node_indices_dict"],
        objective_node_name=simple_dag_and_data["objective_node_name"],
    )
    assert model is not None
    assert "y1" in model.non_root_nodes
    assert "x1" in model.root_nodes
    assert isinstance(
        model.model.node_mlls_dict["y1"], gpytorch.mlls.MarginalLogLikelihood
    )


def test_pogpn_pathwise_fit_and_posterior(simple_dag_and_data):
    """Test fitting the POGPNPathwise model and getting a posterior."""
    with gpytorch.settings.num_likelihood_samples(8):
        model = POGPNPathwise(
            dag=simple_dag_and_data["dag"],
            data_dict=simple_dag_and_data["data_dict"],
            root_node_indices_dict=simple_dag_and_data["root_node_indices_dict"],
            objective_node_name=simple_dag_and_data["objective_node_name"],
        )

        # Test fit
        model.fit(
            data_dict=simple_dag_and_data["data_dict"],
            optimizer="torch",
            maxiter=2,
            lr=0.01,
        )

        # Test posterior
        test_x = torch.randn(5, 2, dtype=torch.float64)
        posterior = model.posterior(test_x)

        # Test rsample
        samples = posterior.rsample(torch.Size([32]))
        assert samples.shape == (32, 5, 1)

        # Test mean and variance from samples
        mean = samples.mean(dim=0)
        variance = samples.var(dim=0)
        assert mean.shape[-2:] == (5, 1)
        assert variance.shape[-2:] == (5, 1)
