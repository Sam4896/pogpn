import gpytorch
import torch

from pogpn import POGPNPathwise, POGPNNodewise, DAG, RootNode, RegressionNode
from pogpn.synthetic_test_function import Ackley
from pogpn.synthetic_test_function.base.dag_experiment_base import (
    DAGSyntheticTestFunction,
)
from pogpn.utils import convert_tensor_to_dict


def build_ackley_dag() -> DAG:
    """Build the DAG for the Ackley function.

    The DAG is a tree with the following structure:
    x -> {y1, y2} -> y3
    """
    x_node = RootNode(name="x", parents=[], node_output_dim=2)
    y1_node = RegressionNode(name="y1", parents=[x_node], node_output_dim=1)
    y2_node = RegressionNode(name="y2", parents=[x_node], node_output_dim=1)
    y3_node = RegressionNode(name="y3", parents=[y1_node, y2_node], node_output_dim=1)
    return DAG(dag_nodes=[x_node, y1_node, y2_node, y3_node])


def generate_data(n: int = 15):
    """Generate data for the Ackley function.

    The data is a dictionary with the following keys: "inputs", "y1", "y2", "y3".
    """
    # Ackley simulator provides dict outputs and a helper for Sobol inputs
    sim_env = Ackley(dim=2, process_stochasticity_std=0.1, observation_noise_std=0.1)
    X = sim_env.get_sobol_samples(n)
    outputs = sim_env(X)
    return sim_env, {
        "inputs": X,
        "y1": outputs["y1"],
        "y2": outputs["y2"],
        "y3": outputs["y3"],
    }


def train_pathwise(
    data_dict, dag, sim_env: DAGSyntheticTestFunction
) -> tuple[POGPNPathwise, list[float]]:
    """Train the pathwise model.

    You can train the whole network jointly or with coordinate-descent (CD).
    - Joint: model.fit(optimizer="torch"|"scipy", ...)
    - CD:   model.fit_torch_with_cd(...)
    """
    # - Joint: model.fit(optimizer="torch"|"scipy", ...)
    # - CD:   model.fit_torch_with_cd(...)
    loss_history = []
    with gpytorch.settings.num_likelihood_samples(32):
        model = POGPNPathwise(
            dag=dag,
            data_dict=data_dict,
            root_node_indices_dict=sim_env.root_node_indices_dict,
            objective_node_name="y3",
            inducing_point_ratio=1.0,
            mll_beta=1.0,
            mll_type="ELBO",
        )

        # Coordinate-descent across nodes (Adam under the hood)
        model.fit_torch_with_cd(lr=2e-2, maxiter=200, loss_history=loss_history)

        # Alternatively: joint training (uncomment one)
        # model.fit(optimizer="torch", lr=1e-2, maxiter=200, loss_history=loss_history)
        # model.fit(optimizer="scipy", maxiter=500, loss_history=loss_history)

    return model, loss_history


def train_nodewise(data_dict, dag, sim_env: DAGSyntheticTestFunction) -> POGPNNodewise:
    """Train the nodewise model.

    Node-wise training fits each node conditionally in topo order.
    """
    with gpytorch.settings.num_likelihood_samples(32):
        model = POGPNNodewise(
            dag=dag,
            data_dict=data_dict,
            root_node_indices_dict=sim_env.root_node_indices_dict,
            objective_node_name="y3",
            inducing_point_ratio=1.0,
            mll_beta=1.0,
            mll_type="ELBO",  # "PLL" is also supported
        )
        model.fit(optimizer="torch", lr=1e-2)
    return model


if __name__ == "__main__":
    dag = build_ackley_dag()
    sim_env, data_dict = generate_data(n=15)

    # Train pathwise (CD shown by default)
    pw_model, pw_loss = train_pathwise(data_dict, dag, sim_env)

    # Optionally: train nodewise instead
    # nw_model = train_nodewise(data_dict, dag, sim_env)

    with torch.no_grad():
        posterior = pw_model.posterior(data_dict["inputs"])
        samples = posterior.rsample(
            sample_shape=torch.Size([64])
        )  # samples for all outputs of the DAG as combined tensor
        print("posterior samples shape:", samples.shape)
        print(
            "posterior samples:",
            convert_tensor_to_dict(samples, {"y1": [0], "y2": [1], "y3": [2]}),
        )
