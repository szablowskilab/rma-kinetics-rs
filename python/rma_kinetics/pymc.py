"""PyTensor/PyMC integration helpers for RMA kinetics."""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import pytensor.tensor as pt
    from pytensor.graph.op import Op
except ImportError as exc:  # pragma: no cover - exercised only without optional deps
    raise ImportError(
        "rma_kinetics.pymc requires the optional PyTensor/PyMC dependencies. "
        "Install with `pip install rma-kinetics[pymc]`."
    ) from exc


class InferenceOp(Op):
    """PyTensor Op for plasma RMA predictions.

    Parameters
    ----------
    solver
        A ``rma_kinetics.models.constitutive.InferenceSolver`` instance.
    """

    itypes = [pt.dvector]
    otypes = [pt.dvector]

    def __init__(self, solver: Any) -> None:
        self.solver = solver

    def perform(self, node: Any, inputs: list[Any], outputs: list[Any]) -> None:
        (log_theta,) = inputs
        log_theta = np.asarray(log_theta, dtype=np.float64)
        outputs[0][0] = np.asarray(self.solver.predict(log_theta), dtype=np.float64)

    def grad(self, inputs: list[Any], output_grads: list[Any]) -> list[Any]:
        (log_theta,) = inputs
        (g_mu,) = output_grads
        return [InferenceVJPOp(self.solver)(log_theta, g_mu)]

    def infer_shape(
        self,
        fgraph: Any,
        node: Any,
        input_shapes: list[tuple[Any, ...]],
    ) -> list[tuple[int]]:
        return [(self.solver.n_obs,)]


class PopulationInferenceOp(Op):
    """PyTensor Op for population plasma RMA predictions."""

    itypes = [pt.dvector, pt.dscalar, pt.dscalar]
    otypes = [pt.dvector]

    def __init__(self, solver: Any) -> None:
        self.solver = solver

    def perform(self, node: Any, inputs: list[Any], outputs: list[Any]) -> None:
        log_prod_mouse, log_bbb, log_deg = inputs
        log_prod_mouse = np.asarray(log_prod_mouse, dtype=np.float64)
        log_bbb = float(np.asarray(log_bbb, dtype=np.float64).item())
        log_deg = float(np.asarray(log_deg, dtype=np.float64).item())
        outputs[0][0] = np.asarray(
            self.solver.predict(log_prod_mouse, log_bbb, log_deg), dtype=np.float64
        )

    def grad(self, inputs: list[Any], output_grads: list[Any]) -> list[Any]:
        log_prod_mouse, log_bbb, log_deg = inputs
        (g_mu,) = output_grads
        grad_prod, grad_bbb, grad_deg = PopulationInferenceVJPOp(self.solver)(
            log_prod_mouse, log_bbb, log_deg, g_mu
        )
        return [grad_prod, grad_bbb, grad_deg]

    def infer_shape(
        self,
        fgraph: Any,
        node: Any,
        input_shapes: list[tuple[Any, ...]],
    ) -> list[tuple[int]]:
        return [(self.solver.n_obs,)]


class PopulationInferenceVJPOp(Op):
    """PyTensor Op for population plasma RMA adjoint VJPs."""

    itypes = [pt.dvector, pt.dscalar, pt.dscalar, pt.dvector]
    otypes = [pt.dvector, pt.dscalar, pt.dscalar]

    def __init__(self, solver: Any) -> None:
        self.solver = solver

    def perform(self, node: Any, inputs: list[Any], outputs: list[Any]) -> None:
        log_prod_mouse, log_bbb, log_deg, g_mu = inputs
        log_prod_mouse = np.asarray(log_prod_mouse, dtype=np.float64)
        log_bbb = float(np.asarray(log_bbb, dtype=np.float64).item())
        log_deg = float(np.asarray(log_deg, dtype=np.float64).item())
        g_mu = np.asarray(g_mu, dtype=np.float64)
        _, grad_prod, grad_bbb, grad_deg = self.solver.predict_and_vjp(
            log_prod_mouse, log_bbb, log_deg, g_mu
        )
        outputs[0][0] = np.asarray(grad_prod, dtype=np.float64)
        outputs[1][0] = np.asarray(grad_bbb, dtype=np.float64)
        outputs[2][0] = np.asarray(grad_deg, dtype=np.float64)

    def infer_shape(
        self,
        fgraph: Any,
        node: Any,
        input_shapes: list[tuple[Any, ...]],
    ) -> list[tuple[Any, ...]]:
        return [(self.solver.n_mice,), (), ()]


class InferenceVJPOp(Op):
    """PyTensor Op for plasma RMA adjoint VJPs."""

    itypes = [pt.dvector, pt.dvector]
    otypes = [pt.dvector]

    def __init__(self, solver: Any) -> None:
        self.solver = solver

    def perform(self, node: Any, inputs: list[Any], outputs: list[Any]) -> None:
        log_theta, g_mu = inputs
        log_theta = np.asarray(log_theta, dtype=np.float64)
        g_mu = np.asarray(g_mu, dtype=np.float64)
        _, grad = self.solver.predict_and_vjp(log_theta, g_mu)
        outputs[0][0] = np.asarray(grad, dtype=np.float64)

    def infer_shape(
        self,
        fgraph: Any,
        node: Any,
        input_shapes: list[tuple[Any, ...]],
    ) -> list[tuple[int]]:
        return [(3,)]


__all__ = [
    "InferenceOp",
    "InferenceVJPOp",
    "PopulationInferenceOp",
    "PopulationInferenceVJPOp",
]
