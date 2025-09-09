# --------------------------------------------------------------------------
#  torch_fit_with_scheduler.py
# --------------------------------------------------------------------------
from __future__ import annotations

from typing import Any, Callable, List, Optional, TYPE_CHECKING

from torch import Tensor

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


def get_loss_closure(
    mll: MarginalLogLikelihood,
    loss_history: Optional[List[float]] = None,
    **kwargs: Any,
) -> Callable[[], Tensor]:
    """Get the loss closure for the GPyTorch model."""

    def closure(**kwargs: Any) -> Tensor:
        model_output = mll.model(*mll.model.train_inputs)
        loss = mll(model_output, mll.model.train_targets, **kwargs)
        if loss_history is not None:
            loss_history.append(-loss.item())  # type: ignore
        return -loss  # type: ignore

    return closure


def fit_custom_torch(
    mll: MarginalLogLikelihood,
    loss_history: Optional[List[float]] = None,
    lr: float = 1e-2,
    maxiter: Optional[int] = None,
):
    """Fit the GPyTorch model using the custom loss closure."""

    def get_loss_closure_with_grads(
        mll: MarginalLogLikelihood,
        parameters: dict[str, Tensor],
        backward: Callable[[Tensor], None] = Tensor.backward,
        reducer: Callable[[Tensor], Tensor] | None = Tensor.sum,
        context_manager: Callable | None = None,
        **kwargs: Any,
    ):
        loss_closure = get_loss_closure(mll, loss_history=loss_history, **kwargs)
        return ForwardBackwardClosure(
            forward=loss_closure,
            backward=backward,
            parameters=parameters,
            reducer=reducer,
            context_manager=context_manager,  # type: ignore
        )

    bounds = None
    _parameters, _bounds = get_parameters_and_bounds(mll)
    bounds = _bounds if bounds is None else {**_bounds, **bounds}

    parameters = {n: p for n, p in _parameters.items() if p.requires_grad}

    fit_gpytorch_mll(
        mll,
        closure=get_loss_closure_with_grads(
            mll=mll,
            parameters=parameters,
        ),
        optimizer=fit_gpytorch_mll_torch,
        optimizer_kwargs={
            "optimizer": optimizer_factory(lr=lr),
            "step_limit": maxiter,
            # "stopping_criterion": None,
        },
    )
