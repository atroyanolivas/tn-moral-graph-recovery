# tn_causal/optimizer.py
from __future__ import annotations
from typing import Protocol, runtime_checkable, Callable, Union
import torch

from .fctn import FCTN


# Type alias: beta can be a float or a callable (step: int) -> float
BetaType = Union[float, Callable[[int], float]]


@runtime_checkable
class InnerOptimizer(Protocol):
    def optimize(
        self,
        tn: FCTN,
        p: torch.Tensor,
        beta: BetaType,
        max_iter: int = 3000,
        tol: float = 1e-6,
    ) -> dict[str, list[float]]:
        ...

    def step(self, tn: FCTN, p: torch.Tensor, beta: float) -> torch.Tensor:
        ...


class GradientInnerOptimizer:
    """Gradient-based inner optimizer using Adam (or L-BFGS).

    Supports beta annealing: pass a callable (step: int) -> float as ``beta``
    to ``optimize()`` and the beta value will be updated at each iteration.
    """

    def __init__(
        self,
        lr: float = 1e-2,
        optimizer_cls: type = torch.optim.Adam,
        **opt_kwargs,
    ) -> None:
        self.lr = lr
        self.optimizer_cls = optimizer_cls
        self.opt_kwargs = opt_kwargs
        self._optimizer: torch.optim.Optimizer | None = None
        self._tn_id: int | None = None

    def _get_optimizer(self, tn: FCTN) -> torch.optim.Optimizer:
        if self._optimizer is None or id(tn) != self._tn_id:
            self._optimizer = self.optimizer_cls(
                list(tn.parameters()), lr=self.lr, **self.opt_kwargs
            )
            self._tn_id = id(tn)
        return self._optimizer

    def reset(self) -> None:
        self._optimizer = None
        self._tn_id = None

    def step(self, tn: FCTN, p: torch.Tensor, beta: float) -> torch.Tensor:
        opt = self._get_optimizer(tn)
        if isinstance(opt, torch.optim.LBFGS):
            def closure() -> torch.Tensor:
                opt.zero_grad()
                loss = tn.loss(p, beta)
                loss.backward()
                return loss
            loss = opt.step(closure)
        else:
            opt.zero_grad()
            loss = tn.loss(p, beta)
            loss.backward()
            opt.step()
        return loss

    def optimize(
        self,
        tn: FCTN,
        p: torch.Tensor,
        beta: BetaType,
        max_iter: int = 3000,
        tol: float = 1e-6,
        verbose: bool = False,
    ) -> dict[str, list[float]]:
        """Run the full optimization loop.

        Parameters
        ----------
        beta : float or callable
            If float, constant beta throughout optimization.
            If callable, must accept an integer step index and return a float.
            This enables beta annealing schedules (see tn_causal.schedules).
        """
        self.reset()
        history: dict[str, list[float]] = {
            "loss": [], "recon": [], "penalty": [], "beta": []
        }

        for i in range(max_iter):
            current_beta = beta(i) if callable(beta) else beta

            loss = self.step(tn, p, current_beta)

            with torch.no_grad():
                recon = tn.reconstruction_loss(p).item()
                penalty = tn.penalty().item()

            history["loss"].append(loss.item())
            history["recon"].append(recon)
            history["penalty"].append(penalty)
            history["beta"].append(current_beta)

            if verbose and (i % 500 == 0 or i == max_iter - 1):
                print(
                    f"  [step {i:5d}]  beta={current_beta:.4e}  "
                    f"loss={loss.item():.4e}  recon={recon:.4e}  "
                    f"penalty={penalty:.4e}"
                )

            if recon < tol:
                break

        return history