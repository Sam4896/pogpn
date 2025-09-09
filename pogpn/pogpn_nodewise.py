from torch import Tensor
from .dag import RegressionNode
from .pogpn_posterior import POGPNPosterior
from .pogpn_base import POGPNBase
import logging
from typing import Optional, Tuple
from botorch.models.model import Model
from gpytorch.mlls import MarginalLogLikelihood
from gpytorch.mlls import VariationalELBO, PredictiveLogLikelihood
from botorch.models.utils.inducing_point_allocators import InducingPointAllocator
from botorch.fit import fit_gpytorch_mll_scipy, fit_gpytorch_mll_torch
from .loss_closure_torch import optimizer_factory
from .custom_approximate_gp import BoTorchVariationalGP
from .likelihood_prior import get_lognormal_likelihood_prior
from botorch.models.transforms.outcome import OutcomeTransform
from gpytorch.likelihoods import GaussianLikelihood, MultitaskGaussianLikelihood
from gpytorch.constraints import GreaterThan
from botorch.models.utils.gpytorch_modules import MIN_INFERRED_NOISE_LEVEL

logger = logging.getLogger("POGPN NODEWISE")

# TODO: Add the minibatching to the custom scipy and torch fit because with a small batch size,
# the intermediate layer MC samples will be better space filling for the given MC sample shape.


class POGPNNodewise(POGPNBase):
    """POGPNNodewise with node-wise conditional training of the nodes."""

    def _get_node_model_and_mll(
        self,
        node_name: str,
        train_X_node: Tensor,  # noqa: N803
        train_Y_node: Tensor,  # noqa: N803
        inducing_point_ratio: float,
        mll_beta: float,
        mll_type: str = "ELBO",
        use_rbf_kernel: bool = True,
        inducing_point_allocator: Optional[InducingPointAllocator] = None,
        learn_inducing_points: bool = True,
        node_observation_noise: Optional[float] = None,
        node_transform: Optional[OutcomeTransform] = None,
    ) -> Tuple[Model, MarginalLogLikelihood]:
        """Get the node model and marginal log likelihood for the given node.

        Args:
            node_name: Name of the node.
                Can be used in case you want to use different models or mlls for different nodes.
                Like: if node_name == 'y1', then the model will be a SingleTaskVariationalGP.
                If node_name == 'y2', then the model will be a MultiTaskVariationalGP.

                Or
                if node_name == 'y1', then the mll will be a VariationalELBO.
                If node_name == 'y2', then the mll will be a PredictiveLogLikelihood.

            train_X_node: Training data for the node.
            train_Y_node: Training labels for the node.
            inducing_point_ratio: The ratio of inducing points to the number of observations.
                This is used to determine the number of inducing points for each node.
            mll_beta: The beta parameter for the ApproximateMarginalLogLikelihood.
            mll_type: The type of marginal log likelihood to use.
                Can be "ELBO" for Variational ELBO or "PLL" for Predictive Log Likelihood.
            use_rbf_kernel: Whether to use an `RBFKernel`. If False, uses a `MaternKernel`.
            inducing_point_allocator: The inducing point allocator for the node.
                This has been provided to be able to use different inducing point allocators for different nodes like GreedyImprovementReduction.
            learn_inducing_points: Whether to learn the inducing points.
            node_observation_noise: The noise level of the node observation.

        Returns:
            Tuple[Model, MarginalLogLikelihood]: The node model and marginal log likelihood.

        """
        likelihood_prior = get_lognormal_likelihood_prior(
            node_observation_noise=node_observation_noise,
            node_transform=node_transform,
        )
        if train_Y_node.shape[-1] == 1:
            likelihood = GaussianLikelihood(
                noise_prior=likelihood_prior,
                noise_constraint=GreaterThan(
                    MIN_INFERRED_NOISE_LEVEL,
                    transform=None,  # type: ignore
                    initial_value=likelihood_prior.mode,
                ),
            )
        elif train_Y_node.shape[-1] > 1:
            likelihood = MultitaskGaussianLikelihood(
                num_tasks=train_Y_node.shape[-1],
                noise_prior=likelihood_prior,
                noise_constraint=GreaterThan(
                    MIN_INFERRED_NOISE_LEVEL,
                    transform=None,  # type: ignore
                    initial_value=likelihood_prior.mode,
                ),
            )
        else:
            likelihood = None

        model = BoTorchVariationalGP(
            train_X=train_X_node,
            train_Y=train_Y_node,
            inducing_points=int(inducing_point_ratio * train_X_node.shape[-2]),
            num_outputs=train_Y_node.shape[-1],
            use_rbf_kernel=use_rbf_kernel,
            inducing_point_allocator=inducing_point_allocator,
            learn_inducing_points=learn_inducing_points,
            likelihood=likelihood,
        )
        if mll_type.upper() == "ELBO":
            mll = VariationalELBO(
                likelihood=model.likelihood,
                model=model.model,
                num_data=self.num_observations,
                beta=mll_beta,
            )
        elif mll_type.upper() == "PLL":
            mll = PredictiveLogLikelihood(
                likelihood=model.likelihood,
                model=model.model,
                num_data=self.num_observations,
                beta=mll_beta,
            )
        return model, mll

    def fit(
        self,
        optimizer: str,
        lr: float = 1e-2,
        maxiter: Optional[int] = None,
    ):
        """Train the POGPNNodewise model."""
        if optimizer == "scipy":
            self.fit_gpytorch_mll_custom_scipy(maxiter=maxiter)
        elif optimizer == "torch":
            self.fit_gpytorch_mll_custom_torch(lr=lr, maxiter=maxiter)
        else:
            raise ValueError(f"Invalid optimizer: {optimizer}")

    def fit_gpytorch_mll_custom_scipy(
        self,
        maxiter: Optional[int] = None,
    ):
        """Train the POGPNNodewise model."""
        optimizer = fit_gpytorch_mll_scipy
        optimizer_kwargs = {
            "options": {"maxiter": maxiter} if maxiter is not None else None,
        }
        self.model.train_pogpn_nodewise(
            optimizer=optimizer, optimizer_kwargs=optimizer_kwargs
        )

    def fit_gpytorch_mll_custom_torch(
        self,
        lr: float = 1e-2,
        maxiter: Optional[int] = None,
    ):
        """Train the POGPNNodewise model."""
        optimizer = fit_gpytorch_mll_torch
        optimizer_kwargs = {
            "optimizer": optimizer_factory(lr=lr),
            "step_limit": maxiter,
        }
        self.model.train_pogpn_nodewise(
            optimizer=optimizer, optimizer_kwargs=optimizer_kwargs
        )

    def forward(self, X: Tensor):  # noqa: N803
        """Forward pass of the POGPNNodewise model."""
        node_models_dict = {}
        for node_name in self.non_root_nodes:
            node = self.dag_nodes[node_name]
            if isinstance(node, RegressionNode) and node.node_model is not None:
                node_models_dict[node_name] = node.node_model

        return POGPNPosterior(
            node_models_dict=node_models_dict,
            X=X,
            dag=self.dag,
            dag_nodes=self.dag_nodes,
            non_root_nodes=self.non_root_nodes,
            root_nodes=self.root_nodes,
            deep_nodes=self.deep_nodes,
            node_parents_dict=self.node_parents_dict,
            root_node_indices_dict=self.root_node_indices_dict,
            objective_node_name=self.objective_node_name,
            posterior_transform=None,
        )
