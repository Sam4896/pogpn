"""Torch optimizer helper utilities and closures used by POGPN.

This module exposes helpers to build closures compatible with BoTorch's
fit_gpytorch_mll_torch optimizer loop. It also supports an optional coordinate
descent mode that restricts gradient accumulation to a single node per
optimizer iteration by overriding the closure's backward behavior.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple, TYPE_CHECKING

from torch import Tensor
import torch

from botorch.fit import fit_gpytorch_mll, fit_gpytorch_mll_torch
from botorch.optim.closures.core import ForwardBackwardClosure
from botorch.optim.utils.model_utils import get_parameters_and_bounds
import logging
from torch.optim import Adam, Optimizer

if TYPE_CHECKING:
    from gpytorch.mlls import MarginalLogLikelihood


logger = logging.getLogger("Torch Optimizer")

# TODO: Minibatching to be implemented


def optimizer_factory(lr: float = 1e-3) -> Callable[[List[Tensor]], Optimizer]:
    """Create a partial function for creating an optimizer."""

    def optimizer_partial(parameters: List[Tensor]) -> Optimizer:
        return Adam(parameters, lr=lr)

    return optimizer_partial


def _build_forward_backward_closure(
    mll: "MarginalLogLikelihood",
    parameters: dict[str, Tensor],
    loss_history: Optional[List[float]],
    node_order: Optional[List[str]],
    cd_state: dict,
) -> ForwardBackwardClosure:
    """Build a ForwardBackwardClosure with optional coordinate descent.

    - If node_order is provided, each iteration updates only the parameters
      of the current node (rotating deterministically).
    - The active node name is passed into the MLL via `coordinate_descent_node`.
    """
    use_cd = bool(node_order)
    if use_cd and "iter" not in cd_state:
        cd_state["iter"] = 0
        cd_state["trace"] = {}

    def current_node() -> Optional[str]:
        if not use_cd:
            return None
        assert node_order is not None
        return node_order[cd_state["iter"] % len(node_order)]

    def select_params_for(node: str) -> List[Tuple[str, Tensor]]:
        # Exact top-level block match to avoid collisions (e.g., y1 vs y11)
        block = f"model.node_model_{node}."
        return [(name, p) for name, p in parameters.items() if name.startswith(block)]

    def forward() -> Tensor:
        # node = current_node()
        node = None
        model_output = mll.model(*mll.model.train_inputs, coordinate_descent_node=node)
        loss = mll(
            model_output,
            mll.model.train_targets,
            coordinate_descent_node=node,
        )
        if loss_history is not None:
            loss_history.append(-loss.item())
        return -loss

    def backward(loss: Tensor) -> None:
        if not use_cd:
            Tensor.backward(loss)
        else:
            node = current_node()
            selected_param_names, selected_params = zip(
                *select_params_for(node), strict=True
            )
            selected_param_names = list(selected_param_names)
            selected_params = list(selected_params)
            # Minimal constant-size debug info
            cd_state["trace"][cd_state["iter"]] = {
                "len_selected_params": len(selected_params),
                "selected_param_names": selected_param_names,
            }
            torch.autograd.backward(loss, inputs=selected_params)
            cd_state["iter"] += 1

    return ForwardBackwardClosure(
        forward=forward,
        parameters=parameters,
        backward=backward,
        reducer=Tensor.sum,
        context_manager=None,
    )


def fit_custom_torch(
    mll: MarginalLogLikelihood,
    loss_history: Optional[List[float]] = None,
    lr: float = 1e-2,
    maxiter: Optional[int] = None,
    *,
    node_order: Optional[List[str]] = None,
    cd_state: Optional[dict] = None,
    stopping_criterion: Optional[Callable[[Tensor], bool]] = None,
):
    """Fit the GPyTorch model using the custom loss closure.

    This supports optional coordinate-descent (CD) via the closure's backward.
    When `node_order` is provided, each optimizer iteration only
    accumulates gradients for parameters who belong to the node in the current iteration of the
    node_order (rotating deterministically through the list).

    Args:
        mll: The MarginalLogLikelihood to optimize.
        loss_history: Optional list to collect per-iteration loss values.
        lr: Learning rate for Adam.
        maxiter: Optional step limit for BoTorch's torch optimizer.
        node_order: Topological ordered list of node-names.
        cd_state: Mutable dict to store rotation state across closure calls. If
            None, an internal dict is created.
        stopping_criterion: Optional stopping criterion for the optimizer.

    """
    if cd_state is None:
        cd_state = {}

    bounds = None
    _parameters, _bounds = get_parameters_and_bounds(mll)
    bounds = _bounds if bounds is None else {**_bounds, **bounds}

    parameters = {n: p for n, p in _parameters.items() if p.requires_grad}

    closure = _build_forward_backward_closure(
        mll=mll,
        parameters=parameters,
        loss_history=loss_history,
        node_order=node_order,
        cd_state=cd_state,
    )

    optimizer_kwargs = {
        "optimizer": optimizer_factory(lr=lr),
        "step_limit": maxiter,
    }

    if stopping_criterion is not None:
        optimizer_kwargs["stopping_criterion"] = stopping_criterion

    fit_gpytorch_mll(
        mll,
        closure=closure,
        optimizer=fit_gpytorch_mll_torch,
        optimizer_kwargs=optimizer_kwargs,
    )
