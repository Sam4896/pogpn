from typing import List

from botorch.fit import fit_gpytorch_mll, fit_gpytorch_mll_scipy

from collections.abc import Callable
from typing import Any, Optional

from botorch.optim.closures.core import ForwardBackwardClosure
from gpytorch.mlls import MarginalLogLikelihood
from torch import Tensor
from botorch.optim.utils.model_utils import get_parameters_and_bounds

# TODO: Minibatching to be implemented


def fit_custom_scipy(
    mll: MarginalLogLikelihood,
    loss_history: Optional[List[float]] = None,
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
        optimizer=fit_gpytorch_mll_scipy,
        optimizer_kwargs={"options": {"maxiter": maxiter}}
        if maxiter is not None
        else None,
    )


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
