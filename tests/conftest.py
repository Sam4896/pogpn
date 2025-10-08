"""Pytest configuration and shared fixtures for POGPN tests."""

import pytest
import torch
from pogpn.dag import DAG, RootNode, RegressionNode


@pytest.fixture
def simple_dag_and_data():
    """Create a simple DAG (x1 -> y1) and corresponding data for testing."""
    # 1. Define Nodes
    x1 = RootNode(name="x1", parents=[], node_output_dim=2)
    y1 = RegressionNode(name="y1", parents=[x1], node_output_dim=1)

    # 2. Create DAG
    dag = DAG([x1, y1])

    # 3. Create Data
    n_samples = 20
    torch.manual_seed(42)
    train_x = torch.randn(n_samples, 2, dtype=torch.float64)
    # Create some relationship between x and y
    train_y = (torch.sin(train_x[:, 0]) + torch.cos(train_x[:, 1])).unsqueeze(-1)

    data_dict = {"inputs": train_x, "y1": train_y, "x1": train_x}
    root_node_indices_dict = {"x1": [0, 1]}

    return {
        "dag": dag,
        "data_dict": data_dict,
        "root_node_indices_dict": root_node_indices_dict,
        "objective_node_name": "y1",
    }
