"""Bayesian optimization example using the Ackley function and qLogEI."""

import torch
import gpytorch
from typing import Dict

from botorch.acquisition import qLogExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.acquisition.objective import GenericMCObjective
from botorch.utils.transforms import normalize, unnormalize
from botorch.models.transforms.outcome import Standardize

from pogpn import (
    POGPNPathwise,
    POGPNNodewise,
    DAG,
    RootNode,
    RegressionNode,
)
from pogpn.synthetic_test_function import Ackley
from pogpn.synthetic_test_function.base.dag_experiment_base import (
    DAGSyntheticTestFunction,
)

sim_env_dim = 5  # input dimension of the simulation environment


def build_dag(node_transforms: Dict[str, Standardize] | None = None) -> DAG:
    """Build a 3-stage DAG: x -> {y1, y2} -> y3.

    If provided, attaches per-node `Standardize` transforms to regression nodes
    (matching how training data is standardized).
    """
    x_node = RootNode(name="x", parents=[], node_output_dim=sim_env_dim)
    y1_node = RegressionNode(
        name="y1",
        parents=[x_node],
        node_output_dim=1,
        node_transform=(node_transforms.get("y1") if node_transforms else None),
    )
    y2_node = RegressionNode(
        name="y2",
        parents=[x_node],
        node_output_dim=1,
        node_transform=(node_transforms.get("y2") if node_transforms else None),
    )
    y3_node = RegressionNode(
        name="y3",
        parents=[y1_node, y2_node],
        node_output_dim=1,
        node_transform=(node_transforms.get("y3") if node_transforms else None),
    )
    return DAG(dag_nodes=[x_node, y1_node, y2_node, y3_node])


def initialize_data(
    sim_env: DAGSyntheticTestFunction, n_init: int = 10
) -> Dict[str, torch.Tensor]:
    """Create initial UNTRANSFORMED training data using Sobol samples and simulator outputs.

    Args:
        sim_env: The simulation environment
        n_init: The number of initial data points

    Returns:
        The initial training data

    """
    x = sim_env.get_sobol_samples(n_init)
    outputs = sim_env(x)
    return {
        "inputs": x,
        "y1": outputs["y1"],
        "y2": outputs["y2"],
        "y3": outputs["y3"],
    }


def transform_data_dict(
    sim_env: DAGSyntheticTestFunction, data_dict: Dict[str, torch.Tensor]
) -> tuple[Dict[str, torch.Tensor], Dict[str, Standardize]]:
    """Normalize inputs to [0,1] and standardize each node output.

    Returns a new transformed dict and a dict of fitted Standardize transforms.
    """
    transformed = {
        "inputs": normalize(data_dict["inputs"], bounds=sim_env.bounds),
        "y1": data_dict["y1"],
        "y2": data_dict["y2"],
        "y3": data_dict["y3"],
    }
    node_transforms: Dict[str, Standardize] = {}
    for node_name in ("y1", "y2", "y3"):
        std = Standardize(m=transformed[node_name].shape[-1])
        transformed[node_name], _ = std(transformed[node_name])
        std.eval()
        node_transforms[node_name] = std
    return transformed, node_transforms


def build_model(
    model_type: str,
    dag: DAG,
    transformed_data: Dict[str, torch.Tensor],
    sim_env: DAGSyntheticTestFunction,
) -> POGPNPathwise | POGPNNodewise:
    """Construct and fit a POGPN model (pathwise or nodewise).

    Args:
        model_type: Which POGPN variant to use ("pathwise" or "nodewise").
        dag: The DAG structure.
        transformed_data: Training data with inputs normalized to [0,1] and outputs standardized.
        sim_env: The simulation environment providing bounds and indexing.

    Returns:
        A fitted POGPN model.

    """
    # Fitting options:
    # - Pathwise joint: model.fit(optimizer="torch"|"scipy")
    # - Pathwise CD:    model.fit_torch_with_cd(...)
    # - Nodewise:       model.fit(optimizer="torch"|"scipy")
    if model_type == "pathwise":
        with gpytorch.settings.num_likelihood_samples(32):
            model = POGPNPathwise(
                dag=dag,
                data_dict=transformed_data,
                root_node_indices_dict=sim_env.root_node_indices_dict,
                objective_node_name="y3",
                inducing_point_ratio=1.0,
                mll_beta=1.0,
                mll_type="ELBO",
            )
            # Coordinate-descent for stability on small data
            model.fit_torch_with_cd(lr=1e-2, maxiter=200)
        return model

    elif model_type == "nodewise":
        with gpytorch.settings.num_likelihood_samples(32):
            model = POGPNNodewise(
                dag=dag,
                data_dict=transformed_data,
                root_node_indices_dict=sim_env.root_node_indices_dict,
                objective_node_name="y3",
                inducing_point_ratio=1.0,
                mll_beta=1.0,
            )
            model.fit(optimizer="torch", lr=1e-2)
        return model

    else:
        raise ValueError("model_type must be 'pathwise' or 'nodewise'")


def propose_candidate_with_qlogei(
    model: POGPNPathwise | POGPNNodewise,
    sim_env: DAGSyntheticTestFunction,
    best_f: float,
) -> torch.Tensor:
    """Propose next point by optimizing qLogEI over the objective node y3.

    Args:
        model: The POGPN model to optimize
        sim_env: The simulation environment
        best_f: The current best observed objective value

    Returns:
        The proposed candidate

    """
    # Build qLogEI on the model's posterior over y3 using a GenericMCObjective
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]))

    def network_to_objective_transform(samples, X=None):  # noqa: N803
        # Extract y3 (last output column) from concatenated network samples
        return samples[..., -1]

    objective = GenericMCObjective(network_to_objective_transform)

    acqf = qLogExpectedImprovement(
        model=model, best_f=best_f, sampler=sampler, objective=objective
    )

    # Optimize over unit cube; the model is trained on normalized inputs
    bounds = torch.stack(
        [
            torch.zeros(
                sim_env.dim, dtype=sim_env.bounds.dtype, device=sim_env.bounds.device
            ),
            torch.ones(
                sim_env.dim, dtype=sim_env.bounds.dtype, device=sim_env.bounds.device
            ),
        ]
    )

    candidate, _ = optimize_acqf(
        acq_function=acqf,
        bounds=bounds,
        q=1,
        num_restarts=8,
        raw_samples=256,
        options={"batch_limit": 32, "maxiter": 200},
    )

    return candidate.detach()


def bo_loop(model_type: str = "pathwise", n_init: int = 10, n_iter: int = 10):
    """Run a simple BO loop with qLogEI using the Ackley simulator."""
    sim_env = Ackley(
        dim=sim_env_dim,
        process_stochasticity_std=0.1,
        observation_noise_std=0.1,
    )
    # Maintain raw data separate from transformed training data
    data_raw = initialize_data(sim_env, n_init=n_init)
    transformed_data, node_transforms = transform_data_dict(sim_env, data_raw)

    dag = build_dag(node_transforms=node_transforms)

    model = build_model(
        model_type=model_type,
        dag=dag,
        transformed_data=transformed_data,
        sim_env=sim_env,
    )

    best_f = transformed_data["y3"].max().item()
    untransformed_best = data_raw["y3"].max().item()
    print("init best:", untransformed_best)

    for t in range(n_iter):
        x_new = propose_candidate_with_qlogei(model, sim_env, best_f)
        # Evaluate simulator in the original space
        x_new_un = unnormalize(x_new, bounds=sim_env.bounds)
        out_raw = sim_env(x_new_un)

        # Append to dataset
        data_raw["inputs"] = torch.cat([data_raw["inputs"], x_new_un], dim=-2)
        for k in ("y1", "y2", "y3"):
            data_raw[k] = torch.cat([data_raw[k], out_raw[k]], dim=-2)

        # Refit model (note: you can switch between joint vs CD here)
        transformed_data, node_transforms = transform_data_dict(sim_env, data_raw)
        dag = build_dag(node_transforms=node_transforms)
        model = build_model(
            model_type=model_type,
            dag=dag,
            transformed_data=transformed_data,
            sim_env=sim_env,
        )
        best_f = max(best_f, transformed_data["y3"].max().item())

        untransformed_best = max(untransformed_best, data_raw["y3"].max().item())
        print(f"iter {t + 1}: best={untransformed_best:.4f}")


if __name__ == "__main__":
    bo_loop(model_type="pathwise", n_init=10, n_iter=5)
