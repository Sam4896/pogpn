#! /usr/bin/env python3

r"""
Gaussian Process Network.

Bayesian Optimization of Function Networks
https://proceedings.neurips.cc/paper/2021/hash/792c7b5aae4a79e78aaeda80516ae2ac-Abstract.html

@article{astudillo2021bayesian,
  title={Bayesian optimization of function networks},
  author={Astudillo, Raul and Frazier, Peter},
  journal={Advances in neural information processing systems},
  volume={34},
  pages={14463--14475},
  year={2021}
}

Implementation based on the original implementation of the paper:
https://github.com/RaulAstudillo06/BOFN
"""

import torch
from typing import Any, List
from botorch import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.utils.gpytorch_modules import get_covar_module_with_dim_scaled_prior
from botorch.models.model import Model
from botorch.posteriors import Posterior
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch import Tensor


class GaussianProcessNetwork(Model):  # noqa: D101
    def __init__(
        self,
        train_X,  # noqa: N803
        train_Y,  # noqa: N803
        dag,
        active_input_indices,
        objective_output_index: int,
        node_GPs=None,  # noqa: N803
        optimizer_kwargs=None,
        node_observation_noise=None,
        use_rbf_kernel=False,
    ) -> None:
        self.train_X = train_X
        self.train_Y = train_Y
        self.dag = dag
        self.n_nodes = dag.get_n_nodes()
        self.root_nodes = dag.get_root_nodes()
        self.active_input_indices = active_input_indices
        self.objective_output_index = objective_output_index
        self.optimizer_kwargs = optimizer_kwargs
        self.node_observation_noise = node_observation_noise
        self.use_rbf_kernel = use_rbf_kernel

        if node_GPs is not None:
            self.node_GPs = node_GPs

        else:
            self.node_GPs: List[Any] = [None for k in range(self.n_nodes)]
            self.node_mlls: List[Any] = [None for k in range(self.n_nodes)]

            for k in self.root_nodes:
                if self.active_input_indices is not None:
                    train_X_node_k = train_X[..., self.active_input_indices[k]]  # noqa: N806
                else:
                    train_X_node_k = train_X  # noqa: N806
                train_Y_node_k = train_Y[..., [k]]  # noqa: N806

                covar_module = get_covar_module_with_dim_scaled_prior(
                    ard_num_dims=train_X_node_k.shape[-1],
                    batch_shape=train_X_node_k.shape[:-2],
                    use_rbf_kernel=self.use_rbf_kernel,
                )

                self.node_GPs[k] = SingleTaskGP(
                    train_X=train_X_node_k,
                    train_Y=train_Y_node_k,
                    covar_module=covar_module,
                )
                self.node_mlls[k] = ExactMarginalLogLikelihood(
                    self.node_GPs[k].likelihood, self.node_GPs[k]
                )
                fit_gpytorch_mll(
                    self.node_mlls[k], optimizer_kwargs=self.optimizer_kwargs
                )

            for k in range(self.n_nodes):
                if self.node_GPs[k] is None:
                    aux = train_Y[..., self.dag.get_parent_nodes(k)].clone()

                    train_X_node_k = torch.cat(  # noqa: N806
                        [train_X[..., self.active_input_indices[k]], aux], -1
                    )
                    train_Y_node_k = train_Y[..., [k]]  # noqa: N806

                    covar_module = get_covar_module_with_dim_scaled_prior(
                        ard_num_dims=train_X_node_k.shape[-1],
                        batch_shape=train_X_node_k.shape[:-2],
                        use_rbf_kernel=self.use_rbf_kernel,
                    )

                    self.node_GPs[k] = SingleTaskGP(
                        train_X=train_X_node_k,
                        train_Y=train_Y_node_k,
                        covar_module=covar_module,
                    )
                    self.node_mlls[k] = ExactMarginalLogLikelihood(
                        self.node_GPs[k].likelihood, self.node_GPs[k]
                    )
                    fit_gpytorch_mll(
                        self.node_mlls[k], optimizer_kwargs=self.optimizer_kwargs
                    )

        super().__init__()
        for k in range(len(self.node_GPs)):
            self.add_module(f"node_GPs_{k}", self.node_GPs[k])

    def posterior(self, X: Tensor, observation_noise=False, posterior_transform=None):
        """Compute the posterior over model outputs at the provided points.

        Args:
            X: A `(batch_shape) x q x d`-dim Tensor, where `d` is the dimension
                of the feature space and `q` is the number of points considered
                jointly.
            observation_noise: If True, add the observation noise from the
                likelihood to the posterior. If a Tensor, use it directly as the
                observation noise (must be of shape `(batch_shape) x q`).

        Returns:
            A `GPyTorchPosterior` object, representing a batch of `b` joint
            distributions over `q` points. Includes observation noise if
            specified.

        """
        return MultivariateNormalNetwork(
            node_GPs=self.node_GPs,
            dag=self.dag,
            X=X,
            objective_output_index=self.objective_output_index,
            indices_X=self.active_input_indices,
            posterior_transform=posterior_transform,
        )

    def forward(self, x: Tensor):  # noqa: D102
        return MultivariateNormalNetwork(
            node_GPs=self.node_GPs,
            dag=self.dag,
            X=x,
            objective_output_index=self.objective_output_index,
            indices_X=self.active_input_indices,
        )

    def condition_on_observations(self, X: Tensor, Y: Tensor, **kwargs: Any) -> Model:  # noqa: N803
        r"""Condition the model on new observations.

        Args:
            X: A `batch_shape x n' x d`-dim Tensor, where `d` is the dimension of
                the feature space, `n'` is the number of points per batch, and
                `batch_shape` is the batch shape (must be compatible with the
                batch shape of the model).
            Y: A `batch_shape' x n' x m`-dim Tensor, where `m` is the number of
                model outputs, `n'` is the number of points per batch, and
                `batch_shape'` is the batch shape of the observations.
                `batch_shape'` must be broadcastable to `batch_shape` using
                standard broadcasting semantics. If `Y` has fewer batch dimensions
                than `X`, it is assumed that the missing batch dimensions are
                the same for all `Y`.
            kwargs: Additional keyword arguments to pass to the

        Returns:
            A `Model` object of the same type, representing the original model
            conditioned on the new observations `(X, Y)` (and possibly noise
            observations passed in via kwargs).

        """
        fantasy_models = [None for k in range(self.n_nodes)]

        for k in self.root_nodes:
            if self.active_input_indices is not None:
                X_node_k = X[..., self.active_input_indices[k]]  # noqa: N806
            else:
                X_node_k = X  # noqa: N806
            Y_node_k = Y[..., [k]]  # noqa: N806
            fantasy_models[k] = self.node_GPs[k].condition_on_observations(
                X_node_k, Y_node_k, noise=torch.ones(Y_node_k.shape[1:]) * 1e-4
            )

        for k in range(self.n_nodes):
            if fantasy_models[k] is None:
                aux = Y[..., self.dag.get_parent_nodes(k)].clone()
                for j in range(len(self.dag.get_parent_nodes(k))):
                    aux[..., j] = (
                        aux[..., j] - self.normalization_constant_lower[k][j]
                    ) / (
                        self.normalization_constant_upper[k][j]
                        - self.normalization_constant_lower[k][j]
                    )
                aux_shape = [aux.shape[0]] + [1] * X[
                    ..., self.active_input_indices[k]
                ].ndim
                X_aux = (  # noqa: N806
                    X[..., self.active_input_indices[k]].unsqueeze(0).repeat(*aux_shape)
                )
                X_node_k = torch.cat([X_aux, aux], -1)  # noqa: N806
                Y_node_k = Y[..., [k]]  # noqa: N806
                fantasy_models[k] = self.node_GPs[k].condition_on_observations(
                    X_node_k, Y_node_k, noise=torch.ones(Y_node_k.shape[1:]) * 1e-4
                )

        return GaussianProcessNetwork(
            dag=self.dag,
            train_X=X,
            train_Y=Y,
            active_input_indices=self.active_input_indices,
            objective_output_index=self.objective_output_index,
            node_GPs=fantasy_models,
            optimizer_kwargs=self.optimizer_kwargs,
            node_observation_noise=self.node_observation_noise,
            use_rbf_kernel=self.use_rbf_kernel,
        )


class MultivariateNormalNetwork(Posterior):  # noqa: D101
    def __init__(  # noqa: D107
        self,
        node_GPs,  # noqa: N803
        dag,
        X,  # noqa: N803
        objective_output_index,
        indices_X=None,  # noqa: N803
        posterior_transform=None,
    ):
        self.node_GPs = node_GPs
        self.dag = dag
        self.n_nodes = dag.get_n_nodes()
        self.root_nodes = dag.get_root_nodes()
        self.X = X
        self.objective_output_index = objective_output_index
        self.active_input_indices = indices_X
        self.posterior_transform = posterior_transform

    @property
    def device(self) -> torch.device:
        r"""The torch device of the posterior."""
        return torch.device(self.X.device)

    @property
    def dtype(self) -> torch.dtype:
        r"""The torch dtype of the posterior."""
        return torch.double

    @property
    def event_shape(self) -> torch.Size:
        r"""The event shape (i.e. the shape of a single sample) of the posterior."""
        shape = [self.X.shape[-2], self.n_nodes]
        shape = torch.Size(shape)
        return self.batch_shape + shape  # type: ignore

    @property
    def batch_shape(self) -> torch.Size:
        """Compute the batch shape of the GaussianProcessNetwork posterior."""
        gp_batch_shape = torch.broadcast_shapes(
            *[gp.batch_shape for gp in self.node_GPs]
        )
        X_batch_shape = self.X.shape[:-2]  # noqa: N806
        return torch.broadcast_shapes(gp_batch_shape, X_batch_shape)

    @property
    def base_sample_shape(self) -> torch.Size:
        """Compute the base sample shape of the GaussianProcessNetwork posterior."""
        return self.event_shape

    def _extended_shape(self, sample_shape: torch.Size) -> torch.Size:
        return sample_shape + self.base_sample_shape  # type: ignore

    @property
    def batch_range(self) -> tuple[int, int]:
        r"""The t-batch range.

        This is used in samplers to identify the t-batch component of the
        `base_sample_shape`. The base samples are expanded over the t-batches to
        provide consistency in the acquisition values, i.e., to ensure that a
        candidate produces same value regardless of its position on the t-batch.
        """
        return 0, -2

    def rsample(self, sample_shape=None):
        """Generate samples from the posterior.

        Args:
            sample_shape: The shape of the samples to generate.

        Returns:
            A tensor of shape `sample_shape + event_shape` containing the samples.

        """
        if sample_shape is None:
            sample_shape = torch.Size([])
        base_samples = torch.randn(
            sample_shape + self.base_sample_shape, device=self.device, dtype=self.dtype
        )
        return self.rsample_from_base_samples(sample_shape, base_samples)

    def rsample_from_base_samples(self, sample_shape=None, base_samples=None):
        """Generate samples from the posterior using provided base samples.

        Args:
            sample_shape: The shape of the samples to generate.
            base_samples: The base samples to use for sampling. If None, new base
                samples will be generated.

        Returns:
            A tensor of shape `sample_shape + event_shape` containing the samples.

        """
        if sample_shape is None:
            sample_shape = torch.Size([])
        if base_samples is None:
            base_samples = torch.randn(
                sample_shape + self.base_sample_shape,
                device=self.device,
                dtype=self.dtype,
            )

        nodes_samples = torch.empty(sample_shape + self.event_shape)
        nodes_samples = nodes_samples.to(self.device).to(self.dtype)
        nodes_samples_available = [False for k in range(self.n_nodes)]

        for k in self.root_nodes:
            if self.active_input_indices is not None:
                X_node_k = self.X[..., self.active_input_indices[k]]  # noqa: N806
            else:
                X_node_k = self.X  # noqa: N806
            multivariate_normal_at_node_k = self.node_GPs[k].posterior(X_node_k)
            nodes_samples[..., k] = (
                multivariate_normal_at_node_k.rsample_from_base_samples(
                    sample_shape, base_samples[..., k]
                )[..., 0]
            )
            nodes_samples_available[k] = True

        while not all(nodes_samples_available):
            for k in range(self.n_nodes):
                parent_nodes = self.dag.get_parent_nodes(k)
                if not nodes_samples_available[k] and all(
                    nodes_samples_available[j] for j in parent_nodes
                ):
                    parent_nodes_samples_normalized = nodes_samples[
                        ..., parent_nodes
                    ].clone()
                    for j in range(len(parent_nodes)):
                        parent_nodes_samples_normalized[..., j] = (
                            parent_nodes_samples_normalized[..., j]
                        )
                    X_node_k = self.X[..., self.active_input_indices[k]]  # type: ignore # noqa: N806
                    aux_shape = [sample_shape[0]] + [1] * X_node_k.ndim
                    X_node_k = X_node_k.unsqueeze(0).repeat(*aux_shape)  # noqa: N806
                    X_node_k = torch.cat(  # noqa: N806
                        [X_node_k, parent_nodes_samples_normalized], -1
                    )
                    multivariate_normal_at_node_k = self.node_GPs[k].posterior(X_node_k)
                    nodes_samples[..., k] = (
                        multivariate_normal_at_node_k.rsample_from_base_samples(
                            sample_shape=torch.Size([1]),
                            base_samples=base_samples[..., k].unsqueeze(dim=0),
                        )[0, ..., 0]
                    )
                    nodes_samples_available[k] = True
        return nodes_samples

    def rsample_objective_node(self, sample_shape=None):
        """Generate samples from the posterior for the objective node."""
        samples = self.rsample(sample_shape)
        return samples[..., self.objective_output_index]

    def rsample_objective_node_from_base_samples(
        self, sample_shape=None, base_samples=None
    ):
        """Generate samples from the posterior for the objective node using provided base samples."""
        if sample_shape is None:
            sample_shape = torch.Size([])
        if base_samples is None:
            base_samples = torch.randn(
                sample_shape + self.base_sample_shape,
                device=self.device,
                dtype=self.dtype,
            )

        samples = self.rsample_from_base_samples(sample_shape, base_samples)
        return samples[..., self.objective_output_index]
