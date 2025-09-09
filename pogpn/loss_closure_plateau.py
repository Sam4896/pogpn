from typing import Any, Callable, List, Optional, TYPE_CHECKING
from torch import Tensor
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from botorch.fit import fit_gpytorch_mll, fit_gpytorch_mll_torch
from botorch.optim.closures.core import ForwardBackwardClosure
from botorch.optim.utils.model_utils import get_parameters_and_bounds

if TYPE_CHECKING:
    from gpytorch.mlls import MarginalLogLikelihood

maxsize = 300


def optimizer_factory(lr: float = 1e-3) -> Callable[[List[Tensor]], Optimizer]:
    """Create a partial function for creating an optimizer."""

    def optimizer_partial(parameters: List[Tensor]) -> Optimizer:
        return Adam(parameters, lr=lr)

    return optimizer_partial


class PlateauShim(ReduceLROnPlateau):
    """Shim that stores the metric from the loss closure."""

    def __init__(self, optimizer, **kwargs):
        """Initialize the shim."""
        super().__init__(optimizer, **kwargs)
        self._latest: Optional[float] = None

    def update(self, value: float) -> None:
        """Update the shim with the metric from the loss closure."""
        self._latest = value

    def step(self, epoch: Optional[int] = None) -> None:
        """Step the shim."""
        if self._latest is not None:
            super().step(self._latest, epoch)
            self._latest = None


def get_loss_closure(
    mll,
    scheduler: PlateauShim,
    loss_history=None,
    **kwargs: Any,
):
    """Get the loss closure for the GPyTorch model."""

    def closure(**kwargs: Any):
        out = mll.model(*mll.model.train_inputs)
        loss = mll(out, mll.model.train_targets, **kwargs)
        metric = -loss.item()
        scheduler.update(metric)
        if loss_history is not None:
            loss_history.append(metric)
        return -loss

    return closure


def fit_custom_plateau(
    mll: "MarginalLogLikelihood",
    lr: float = 1e-1,
    maxiter: Optional[int] = None,
    plateau_kwargs: Optional[dict] = None,
    loss_history: Optional[List[float]] = None,
):
    """Fit the GPyTorch model using Adam and ReduceLROnPlateau scheduler."""
    if plateau_kwargs is None:
        plateau_kwargs = {
            "mode": "max",
            "factor": 0.5,
            "patience": 10,
            "threshold": 1e-4,
            "threshold_mode": "rel",
        }
    _parameters, _ = get_parameters_and_bounds(mll)
    parameters = {n: p for n, p in _parameters.items() if p.requires_grad}
    optimizer = Adam(parameters.values(), lr=lr)
    scheduler = PlateauShim(optimizer, **plateau_kwargs)

    def get_loss_closure_with_grads(
        mll: "MarginalLogLikelihood",
        parameters: dict[str, Tensor],
        backward: Callable[[Tensor], None] = Tensor.backward,
        reducer: Callable[[Tensor], Tensor] | None = Tensor.sum,
        context_manager: Callable | None = None,
        **kwargs: Any,
    ):
        loss_closure = get_loss_closure(
            mll, scheduler=scheduler, loss_history=loss_history, **kwargs
        )
        return ForwardBackwardClosure(
            forward=loss_closure,
            backward=backward,
            parameters=parameters,
            reducer=reducer,
            context_manager=context_manager,  # type: ignore
        )

    if maxiter is None:
        maxiter = maxsize

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
            "stopping_criterion": None,
            "scheduler": scheduler,
        },
    )
