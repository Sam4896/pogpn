"""Pathwise POGPN implementation with path-wise conditional training."""

from torch import Tensor
from .pogpn_base import POGPNBase
from .loss_closure_scipy import fit_custom_scipy
from .loss_closure_torch import fit_custom_torch
import logging
from .pogpn_mll import POGPNPathwiseMLL
from typing import Optional, List, Tuple, Any, Callable
from botorch.models.model import Model
from gpytorch.mlls import MarginalLogLikelihood
from .pogpn_mll import (
    VariationalELBOCustom,
    PredictiveLogLikelihoodCustom,
    VariationalELBOCustomWithNaN,
)
from botorch.models.utils.inducing_point_allocators import InducingPointAllocator
from .custom_approximate_gp import BoTorchVariationalGP
from .likelihood_prior import get_lognormal_likelihood_prior
from botorch.models.transforms.outcome import OutcomeTransform
from gpytorch.likelihoods import GaussianLikelihood, MultitaskGaussianLikelihood
from gpytorch.constraints import GreaterThan
from botorch.models.utils.gpytorch_modules import MIN_INFERRED_NOISE_LEVEL
from torch.optim.lr_scheduler import _LRScheduler

logger = logging.getLogger("POGPN PATHWISE")


# TODO: Add the minibatching to the custom scipy and torch fit because with a small batch size,
# the intermediate layer MC samples will be better space filling for the given MC sample shape.


class POGPNPathwise(POGPNBase):
    """POGPNPathwise with node-wise conditional training of the nodes."""

    def _get_node_model_and_mll(
        self,
        node_name: str,
        train_X_node: Tensor,  # noqa: N803
        train_Y_node: Tensor,  # noqa: N803
        inducing_point_ratio: float,
        mll_beta: float,
        mll_type: str = "PLL",
        use_rbf_kernel: bool = True,
        inducing_point_allocator: Optional[InducingPointAllocator] = None,
        learn_inducing_points: bool = True,
        node_observation_noise: Optional[float] = None,
        node_transform: Optional[OutcomeTransform] = None,
    ) -> Tuple[Model, MarginalLogLikelihood]:
        """Get the node model and marginal log likelihood for the given node.

        Args:
            node_name: Name of the node.
                Can be used in case you want to use different models or mlls for different nodes
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
            node_transform: The outcome transform for the node. This can be used to get likelihood prior using noise and the transform of the node.

        Returns:
            Tuple[Model, MarginalLogLikelihood]: The node model and marginal log likelihood.

        """
        if node_observation_noise is not None:
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
            if self.masks_dict is not None and node_name in self.masks_dict:
                mll = VariationalELBOCustomWithNaN(
                    likelihood=model.likelihood,
                    model=model.model,
                    num_data=self.num_observations,
                    beta=mll_beta,
                )
                # Attach row mask for this node
                mll.row_mask = self.masks_dict[node_name]
            else:
                mll = VariationalELBOCustom(
                    likelihood=model.likelihood,
                    model=model.model,
                    num_data=self.num_observations,
                    beta=mll_beta,
                )
        elif mll_type.upper() == "PLL":
            mll = PredictiveLogLikelihoodCustom(
                likelihood=model.likelihood,
                model=model.model,
                num_data=self.num_observations,
                beta=mll_beta,
            )
        return model, mll

    def forward(self, X: Tensor):  # noqa: N803
        """Forward pass of the POGPNPathwise model."""
        return self.model(X)

    def fit(
        self,
        optimizer: str,
        lr: float = 1e-2,
        maxiter: Optional[int] = None,
        loss_history: Optional[List[float]] = None,
        node_output_size_normalization: bool = False,
        **kwargs: Any,
    ):
        """Train the POGPNPathwise model.

        Args:
            optimizer: The optimizer to use for the training.
                Can be "scipy" or "torch".
                If scipy, then LBFGS is used.
                If torch, then Adam is used.
            lr: The learning rate to use for the training.
                Only used if optimizer is "torch".
            maxiter: The maximum number of iterations to train for.
            loss_history: The loss history to store the loss values.
            node_output_size_normalization: Whether to normalize the node output size.
                If True, then the node output size is normalized by the number of observations.
                If False, then the node output size is not normalized.
            kwargs: Additional keyword arguments (currently unused).

        """
        if optimizer == "scipy":
            self.fit_scipy(
                maxiter=maxiter,
                loss_history=loss_history,
                node_output_size_normalization=node_output_size_normalization,
            )
        elif optimizer == "torch":
            self.fit_torch(
                lr=lr,
                maxiter=maxiter,
                loss_history=loss_history,
                node_output_size_normalization=node_output_size_normalization,
            )
        else:
            raise ValueError(f"Invalid optimizer: {optimizer}")

    def fit_scipy(
        self,
        maxiter: Optional[int] = None,
        loss_history: Optional[List[float]] = None,
        node_output_size_normalization: bool = False,
    ):
        """Train the POGPNPathwise model.

        Analysis shows that the KL loss is best minimized and LL is best maximized using scipy optimizer.
        """
        mll = POGPNPathwiseMLL(
            likelihood=self.likelihood,
            model=self.model,
            node_output_size_normalization=node_output_size_normalization,
        ).to(self.device, self.dtype)
        fit_custom_scipy(mll, loss_history, maxiter=maxiter)

    def fit_torch(
        self,
        lr: float = 1e-2,
        maxiter: Optional[int] = None,
        loss_history: Optional[List[float]] = None,
        node_output_size_normalization: bool = False,
    ):
        """Train the POGPNPathwise model."""
        mll = POGPNPathwiseMLL(
            likelihood=self.likelihood,
            model=self.model,
            node_output_size_normalization=node_output_size_normalization,
        ).to(self.device, self.dtype)
        fit_custom_torch(mll, loss_history, lr=lr, maxiter=maxiter)

    def fit_torch_with_cd(
        self,
        lr: float = 1e-2,
        maxiter: Optional[int] = None,
        loss_history: Optional[List[float]] = None,
        node_output_size_normalization: bool = False,
        cd_state: Optional[dict] = None,
        stopping_criterion: Optional[Callable[[Tensor], bool]] = None,
        lr_scheduler: Optional[
            _LRScheduler | Callable[..., _LRScheduler] | None
        ] = None,
    ):
        """Train the POGPNPathwise model using closure-based coordinate descent.

        This reuses BoTorch's torch fit loop and heuristics, while restricting
        each optimizer iteration to update parameters of exactly one node in a
        deterministic topological order via the closure's backward pass.

        Args:
            lr: Learning rate for Adam.
            maxiter: Optional step limit to pass to BoTorch's torch fitter. If
                None, BoTorch controls step limit and early stopping.
            loss_history: Optional list that will be appended with the (negative)
                loss values per iteration by the shared closure.
            node_output_size_normalization: Whether to normalize node output size
                in the POGPNPathwiseMLL.
            cd_state: Optional state to store the CD state.
            stopping_criterion: Optional stopping criterion for the optimizer.
            lr_scheduler: Optional learning rate scheduler (e.g., ReduceLROnPlateau).

        """
        # Build the global MLL as in joint training
        mll = POGPNPathwiseMLL(
            likelihood=self.likelihood,
            model=self.model,
            node_output_size_normalization=node_output_size_normalization,
        ).to(self.device, self.dtype)

        # Deterministic topological order of trainable (non-root) nodes
        node_order = self.dag.get_deterministic_topological_sort_subset(
            self.non_root_nodes
        )

        cd_state: dict = cd_state if cd_state is not None else {}

        fit_custom_torch(
            mll,
            loss_history=loss_history,
            lr=lr,
            maxiter=maxiter,
            node_order=node_order,
            cd_state=cd_state,
            stopping_criterion=stopping_criterion,
            lr_scheduler=lr_scheduler,
        )
        return cd_state
