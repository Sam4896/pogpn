# Partially Observable Gaussian Process Network (POGPN) with BoTorch

This package contains the implementation of [Partially Observable Gaussian Process Networks (POGPN)](https://arxiv.org/abs/2502.13905), a flexible model for structured multi-output regression, integrated with the BoTorch library. It is particularly well-suited for Bayesian Optimization tasks where the process is multi-stage or a process network with noisy or partial intermediate, observable outputs that are causally related.

## Overview

A POGPN is a Directed Acyclic Graph (DAG) where each node represents a random variable modeled by a Gaussian Process (GP). The key idea is that the outputs of parent nodes in the graph become the inputs to their children nodes. This structure allows for modeling complex dependencies and propagating uncertainty through the network.

This implementation uses Variational Inference (VI) to approximate the posterior distribution, making it scalable to larger datasets.

## File-by-File Detailed Explanation

### `dag.py`

**Purpose**: Defines the classes for creating the structure of the Directed Acyclic Graph (DAG).

-   **`DAGNode`**: A dataclass representing a generic node in the DAG. It stores the node's `name`, its `parents` (as a list of `DAGNode` objects), and its `node_output_dim`.
-   **`RootNode`**: A subclass of `DAGNode` specifically for root nodes (nodes with no parents). It enforces that it cannot have parents.
-   **`RegressionNode`**: A subclass of `DAGNode` for non-root nodes that represent a regression output. It includes fields to store the node-specific GP `node_model`, its `node_mll` (Marginal Log-Likelihood), and other training-related attributes.
-   **`ClassificationNode`**: Similar to `RegressionNode`, but for classification tasks.
-   **`DAG`**: A class that inherits from `networkx.DiGraph`. It builds the graph from a sequence of `DAGNode` objects and provides helpful methods to query the graph's structure, such as:
    -   `get_node_parents(node_name)`: Returns the parents of a given node.
    -   `root_nodes`: A property that returns all root nodes.
    -   `get_full_deterministic_topological_sort()`: Returns a topologically sorted list of all nodes, ensuring a deterministic order. This is crucial for iterating through the graph in a causally-correct order.
    -   `plot_dag()`: A utility to visualize the graph structure.

### `pogpn_base.py`

**Purpose**: This is the core of the POGPN model, defining the abstract base class.

-   **`_POGPNModelDict`**: An internal `gpytorch.models.GP` subclass that acts as a container for all the node-specific GP models (`node_models_dict`). It's registered as a `torch.nn.Module` so that PyTorch can track all parameters of the network. Its `forward` method implements the logic for propagating samples through the DAG for pathwise training.
-   **`POGPNBase`**: This is the main abstract class that users will extend.
    -   `__init__`: Initializes the DAG, organizes nodes (root, non-root, deep), prepares data dictionaries, and calls `_initialize_node_models`.
    -   `_initialize_node_models`: Iterates through the non-root nodes and calls the abstract `_get_node_model_and_mll` method to create the GP model and MLL for each one.
    -   `_get_node_model_and_mll`: **Abstract method**. Subclasses must implement this to define how each node's GP model is constructed.
    -   `forward`: **Abstract method**. Subclasses must implement the forward pass.
    -   `posterior`: Returns a `POGPNPosterior` object, which is used for making predictions and sampling.

### `pogpn_nodewise.py`

**Purpose**: This file implements the nodewise training strategy.

-   **`POGPNNodewise`**: A subclass of `POGPNBase`.
    -   `_get_node_model_and_mll`: Implements the abstract method to create a `BoTorchVariationalGP` and a standard BoTorch `VariationalELBO` or `PredictiveLogLikelihood` for each node.
    -   `fit`: Contains the training loop. It iterates through the nodes in topological order. For each node, it calls a standard `fit_gpytorch_mll` function to optimize the parameters of that node's GP. For deep nodes, it first generates input samples by sampling from the posteriors of the parent nodes.
    -   `forward`: Implemented to return a `POGPNPosterior` for inference.

### `pogpn_pathwise.py`

**Purpose**: This file implements the end-to-end pathwise training strategy.

-   **`POGPNPathwise`**: A subclass of `POGPNBase`.
    -   `_get_node_model_and_mll`: Implements the abstract method to create a `BoTorchVariationalGP` and one of the custom MLLs (e.g., `VariationalELBOCustom`) for each node.
    -   `fit`: This is the main training method. It sets up the `POGPNPathwiseMLL` which aggregates the losses from all nodes. It then calls one of the custom fitting functions (`fit_custom_scipy` or `fit_custom_torch`) to optimize all network parameters jointly.
    -   `forward`: The forward pass simply calls the `forward` method of its internal `_POGPNModelDict` to get the network output.

### `pogpn_posterior.py`

**Purpose**: This file defines the posterior distribution for the POGPN.

-   **`POGPNPosterior`**: A subclass of `botorch.posteriors.Posterior`. This is the object returned by the model's `.posterior()` method.
    -   `_rsample_from_base_samples`: This is the core method for inference. It takes a set of base samples (standard normal) and transforms them into samples from the joint posterior of the POGPN. It does this by propagating samples through the graph in topological order.
    -   `rsample`: A convenience method to get samples. It can return all node outputs concatenated into a single tensor.
    -   `rsample_dict`: Returns a dictionary of samples, with node names as keys.
    -   `rsample_objective_node`: A helper to directly get samples for just the specified objective node.

### `pogpn_mll.py`

**Purpose**: This file contains custom Marginal Log-Likelihood classes required for pathwise training.

-   **`_ApproximateMarginalLogLikelihoodCustom`**: An abstract base class for custom approximate MLLs.
-   **`POGPNPathwiseMLL`**: The main MLL used for the pathwise training strategy. Its `forward` method iterates through all node-specific MLLs, calculates their individual losses, and sums them up to get a single loss for the entire network.
-   **`VariationalELBOCustom` / `PredictiveLogLikelihoodCustom`**: These are subclasses of BoTorch's `VariationalELBO` and `PredictiveLogLikelihood`. The key difference is that their `forward` method returns a *dictionary* of loss components (log-likelihood, KL divergence, log-prior) instead of a single scalar loss.
-   **`VariationalELBOCustomWithNaN`**: An extension that can handle missing data (represented as NaNs) by using a mask to exclude those points from the log-likelihood calculation.

### Supporting Modules

-   **`dict_dataset.py`**: Provides custom PyTorch `Dataset` and `DataLoader` to handle data structured as a dictionary.
    -   **`DictDataset`**: A custom `torch.utils.data.Dataset` that takes a dictionary of tensors and allows iteration by index, returning a dictionary slice.
    -   **`DictDataLoader`**: A custom `torch.utils.data.DataLoader` that wraps `DictDataset` for batching and shuffling.

-   **`likelihood_prior.py`**: Provides utilities for setting priors on the likelihood's noise parameter.
    -   **`get_lognormal_likelihood_prior(...)`**: Creates a `gpytorch.priors.LogNormalPrior` for the noise parameter, centered around an expected `node_observation_noise` and accounting for outcome transformations.

-   **`loss_closure_scipy.py` & `loss_closure_torch.py`**: Provide custom wrappers around BoTorch's model fitting functions.
    -   **`fit_custom_scipy(...)`** and **`fit_custom_torch(...)`**: Set up a loss closure and call the appropriate BoTorch `fit_gpytorch_mll` function with either a SciPy (`L-BFGS`) or PyTorch (`Adam`) optimizer.
    -   **`get_loss_closure(...)`**: Helper function that creates the closure called by the optimizer at each step to calculate the negative MLL.

-   **`utils.py`**: Contains various helper functions.
    -   **`convert_tensor_to_dict`**: Splits a single large tensor into a dictionary of smaller tensors based on an index map.
    -   **`convert_dict_to_tensor`**: The inverse of the above; combines a dictionary of tensors into a single large tensor.
    -   **`consolidate_mvn_mixture`**: A function for moment-matching a mixture of Multivariate Normal distributions into a single one.
    -   **`handle_nans_and_create_mask`**: A utility to find NaNs in data, create a boolean mask, and impute the NaN values.

## How to cite

If you use this software, please cite both the software and the research paper, as well as BoTorch and GPyTorch.

**Research Paper (BibTeX):**

```bibtex
@article{kiroriwal2025pogpn,
  title={Partially Observable Gaussian Process Network and Doubly Stochastic Variational Inference},
  author={Kiroriwal, Saksham and Pfrommer, Julius and Beyerer, J{\"u}rgen},
  journal={arXiv preprint arXiv:2502.13905},
  year={2025},
  url={https://arxiv.org/abs/2502.13905}
}
```

**Software Package (BibTeX):**

```bibtex
@software{pogpn_2025,
  author  = {Kiroriwal, Saksham},
  title   = {POGPN: Partially Observable Gaussian Process Networks},
  year    = {2025},
  version = {0.0.1},
  url     = {https://pypi.org/project/pogpn/}
}
```

BoTorch:

```bibtex
@inproceedings{balandat2020botorch,
  title={BoTorch: A Framework for Efficient Monte-Carlo Bayesian Optimization},
  author={Balandat, Maximilian and Karrer, Brian and Jiang, Daniel R. and Daulton, Samuel and Letham, Benjamin and Wilson, Andrew Gordon and Bakshy, Eytan},
  booktitle={NeurIPS},
  year={2020}
}
```

See `CITATION.cff` for citation metadata and `THIRD_PARTY_LICENSES.md` for third‑party licenses.