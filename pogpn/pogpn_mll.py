from abc import ABC, abstractmethod

from typing import Any, List

import torch
from gpytorch.mlls import MarginalLogLikelihood
from gpytorch.mlls._approximate_mll import _ApproximateMarginalLogLikelihood


class _ApproximateMarginalLogLikelihoodCustom(MarginalLogLikelihood, ABC):
    """Custom approximate marginal log likelihood."""

    def __init__(self, likelihood, model):
        """Initialize the _ApproximateMarginalLogLikelihoodCustom."""
        super().__init__(likelihood, model)

    @abstractmethod
    def _log_likelihood_term(self, approximate_dist_f_dict, target_dict, **kwargs):
        """Log likelihood term."""
        raise NotImplementedError

    def forward(
        self,
        approximate_dist_f_dict: dict[str, Any],
        target_dict: dict[str, Any],
        **kwargs,
    ):
        """Forward pass.

        Args:
            approximate_dist_f_dict: Dictionary of approximate distributions for each node.
            target_dict: Dictionary of target values for each node.
            **kwargs: Additional keyword arguments.
                coordinate_descent_node: Optional[str] = None,
                This is the node to perform coordinate descent on. If None, all nodes are used to calculate the loss.

        """
        loss = []

        if kwargs.get("coordinate_descent_node", None) is not None:
            coordinate_descent_node = kwargs["coordinate_descent_node"]
            if isinstance(coordinate_descent_node, str):
                coordinate_descent_node = [coordinate_descent_node]
            elif isinstance(coordinate_descent_node, (list, tuple, set)):
                coordinate_descent_node = coordinate_descent_node
            else:
                raise ValueError(
                    "coordinate_descent_node must be a string or list of strings"
                )
        else:
            coordinate_descent_node = list(self.node_mlls_dict.keys())

        for node_name in coordinate_descent_node:
            if node_name in target_dict.keys():
                loss_dict = self.node_mlls_dict[node_name](
                    approximate_dist_f_dict[node_name],
                    target_dict[node_name],
                    self.node_output_size_dict[node_name],
                )
            else:
                loss_dict = self.node_mlls_dict[node_name](
                    approximate_dist_f_dict[node_name],
                    None,
                    self.node_output_size_dict[node_name],
                )

            for key in loss_dict.keys():
                self.dag_nodes[node_name].node_mll_loss_history[key].append(
                    loss_dict[key].item()  # type: ignore
                )

            loss.append(sum(loss_dict.values()))

        return sum(loss)


class POGPNPathwiseMLL(_ApproximateMarginalLogLikelihoodCustom):  # noqa: D101
    def __init__(self, likelihood, model, node_output_size_normalization: bool = False):
        """Initialize the POGPNPathwiseVariationalELBO."""
        super().__init__(likelihood, model)
        self.node_mlls_dict = model.node_mlls_dict
        self.dag_nodes = model.dag_nodes
        # Store row masks dict for possible downstream usage
        self.row_masks_dict = getattr(model, "masks_dict", None)
        if node_output_size_normalization:
            self.node_output_size_dict = {
                node_name: self.dag_nodes[node_name].node_output_dim
                for node_name in self.node_mlls_dict.keys()
            }
        else:
            self.node_output_size_dict = dict.fromkeys(self.node_mlls_dict.keys(), 1.0)

    def _log_likelihood_term(self, variational_dist_f_dict, target_dict, **kwargs):
        """Log likelihood term."""
        pass

    def forward(self, variational_dist_f_dict: List[Any], target_dict, **kwargs):
        """Forward pass."""
        return super().forward(variational_dist_f_dict, target_dict, **kwargs)


class VariationalELBOCustom(_ApproximateMarginalLogLikelihood):
    """Custom Variational ELBO that allows for logging."""

    def _log_likelihood_term(self, variational_dist_f, target, **kwargs):
        """Log likelihood term."""
        return self.likelihood.expected_log_prob(
            target, variational_dist_f, **kwargs
        ).sum(-1)

    def forward(self, approximate_dist_f, target, node_output_size=1.0, **kwargs):
        """Forward pass."""
        # Get likelihood term and KL term
        num_batch = approximate_dist_f.event_shape[0]
        kl_divergence = self.model.variational_strategy.kl_divergence().div(
            self.num_data / self.beta
        )

        if target is not None:
            log_likelihood = self._log_likelihood_term(
                approximate_dist_f, target, **kwargs
            ).div(num_batch * node_output_size)
            # if log_likelihood.numel() > 1:
            #     log_likelihood = log_likelihood.mean()
        else:
            log_likelihood = torch.zeros_like(kl_divergence)

        # Add any additional registered loss terms
        added_loss = torch.zeros_like(kl_divergence)
        for added_loss_term in self.model.added_loss_terms():
            added_loss.add_(added_loss_term.loss())

        # Log prior term
        log_prior = torch.zeros_like(kl_divergence)
        for name, module, prior, closure, _ in self.named_priors():
            log_prior.add_(prior.log_prob(closure(module)).sum().div(self.num_data))

        return {
            "log_likelihood": log_likelihood,
            "kl_divergence": -kl_divergence,
            "log_prior": log_prior,
            "added_loss": -added_loss,
        }

    def __call__(self, *inputs, **kwargs):
        """Call the VariationalELBOCustom.

        Custom made so as to avoid error of _validate_module_outputs as dict output gives error.
        """
        return self.forward(*inputs, **kwargs)


class VariationalELBOCustomWithNaN(_ApproximateMarginalLogLikelihood):
    """Custom Variational ELBO that allows for logging."""

    def _log_likelihood_term(self, variational_dist_f, target, **kwargs):
        """Log likelihood term."""
        ll = self.likelihood.expected_log_prob(
            target, variational_dist_f, **kwargs
        ).sum(-1)

        # If a row_mask attribute is present, use it to exclude imputed rows
        row_mask = getattr(self, "row_mask", None)
        if row_mask is not None:
            # row_mask == True indicates row was imputed → exclude from loss
            if row_mask.shape != ll.shape:
                row_mask = row_mask.view_as(ll)
            ll = ll * (~row_mask).float()

        return ll

    def forward(self, approximate_dist_f, target, node_output_size=1.0, **kwargs):
        """Forward pass."""
        # Get likelihood term and KL term
        num_batch = approximate_dist_f.event_shape[0]
        kl_divergence = self.model.variational_strategy.kl_divergence().div(
            self.num_data / self.beta
        )

        if target is not None:
            log_likelihood = self._log_likelihood_term(
                approximate_dist_f, target, **kwargs
            ).div(num_batch * node_output_size)
            # if log_likelihood.numel() > 1:
            #     log_likelihood = log_likelihood.mean()
        else:
            log_likelihood = torch.zeros_like(kl_divergence)

        # Add any additional registered loss terms
        added_loss = torch.zeros_like(kl_divergence)
        for added_loss_term in self.model.added_loss_terms():
            added_loss.add_(added_loss_term.loss())

        # Log prior term
        log_prior = torch.zeros_like(kl_divergence)
        for name, module, prior, closure, _ in self.named_priors():
            log_prior.add_(prior.log_prob(closure(module)).sum().div(self.num_data))

        return {
            "log_likelihood": log_likelihood,
            "kl_divergence": -kl_divergence,
            "log_prior": log_prior,
            "added_loss": -added_loss,
        }

    def __call__(self, *inputs, **kwargs):
        """Call the VariationalELBOCustom.

        Custom made so as to avoid error of _validate_module_outputs as dict output gives error.
        """
        return self.forward(*inputs, **kwargs)


class PredictiveLogLikelihoodCustom(_ApproximateMarginalLogLikelihood):
    """Custom Predictive Log Likelihood that allows for logging."""

    def _log_likelihood_term(self, variational_dist_f, target, **kwargs):
        """Log likelihood term."""
        if target is None:
            return torch.zeros(
                variational_dist_f.event_shape[0], device=variational_dist_f.mean.device
            )
        return self.likelihood.log_marginal(target, variational_dist_f, **kwargs).sum(
            -1
        )

    def forward(self, approximate_dist_f, target, node_output_size=1.0, **kwargs):
        """Forward pass."""
        # Get likelihood term and KL term
        num_batch = approximate_dist_f.event_shape[0]
        kl_divergence = self.model.variational_strategy.kl_divergence().div(
            self.num_data / self.beta
        )

        if target is not None:
            log_likelihood = self._log_likelihood_term(
                approximate_dist_f, target, **kwargs
            ).div(num_batch * node_output_size)
            # if log_likelihood.numel() > 1:
            #     log_likelihood = log_likelihood.mean()
        else:
            log_likelihood = torch.zeros_like(kl_divergence)

        # Add any additional registered loss terms
        added_loss = torch.zeros_like(kl_divergence)
        for added_loss_term in self.model.added_loss_terms():
            added_loss.add_(added_loss_term.loss())

        # Log prior term
        log_prior = torch.zeros_like(kl_divergence)
        for name, module, prior, closure, _ in self.named_priors():
            log_prior.add_(prior.log_prob(closure(module)).sum().div(self.num_data))

        return {
            "log_likelihood": log_likelihood,
            "kl_divergence": -kl_divergence,
            "log_prior": log_prior,
            "added_loss": -added_loss,
        }

    def __call__(self, *inputs, **kwargs):
        """Call the VariationalELBOCustom.

        Custom made so as to avoid error of _validate_module_outputs as dict output gives error.
        """
        return self.forward(*inputs, **kwargs)
