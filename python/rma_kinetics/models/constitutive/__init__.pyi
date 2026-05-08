from typing import TYPE_CHECKING

from numpy import float64
from numpy.typing import ArrayLike, NDArray

from . import erasable as erasable

if TYPE_CHECKING:
    from ... import Solution
    from ...solvers import Solver

class InferenceSolver:
    """Constitutive inference solver for plasma RMA predictions and adjoint VJPs."""

    def __init__(
        self,
        obs_time: ArrayLike,
        *,
        init_state: State | None = None,
        t0: float = 0.0,
        tf: float | None = None,
        dt: float = 0.25,
    ) -> None: ...
    @property
    def n_obs(self) -> int: ...
    def predict(self, log_params: ArrayLike) -> NDArray[float64]: ...
    def predict_and_vjp(
        self, log_params: ArrayLike, cotangent: ArrayLike
    ) -> tuple[NDArray[float64], NDArray[float64]]: ...
    def clear_cache(self) -> None: ...

class PopulationInferenceSolver:
    """Constitutive population inference solver for mouse-specific production rates."""

    def __init__(
        self,
        mouse_id: ArrayLike,
        obs_time: ArrayLike,
        n_mice: int,
        *,
        init_state: State | None = None,
        t0: float = 0.0,
        tf: float | None = None,
        dt: float = 0.25,
    ) -> None: ...
    @property
    def n_obs(self) -> int: ...
    @property
    def n_mice(self) -> int: ...
    def predict(
        self, log_prod_mouse: ArrayLike, log_bbb: float, log_deg: float
    ) -> NDArray[float64]: ...
    def predict_and_vjp(
        self,
        log_prod_mouse: ArrayLike,
        log_bbb: float,
        log_deg: float,
        cotangent: ArrayLike,
    ) -> tuple[NDArray[float64], NDArray[float64], float, float]: ...
    def clear_cache(self) -> None: ...

class Model:
    """Constitutive RMA expression model."""

    def __init__(
        self, prod: float = 0.2, bbb_transport: float = 0.6, deg: float = 0.007
    ) -> None: ...
    def solve(
        self, t0: float, tf: float, dt: float, init_state: State, solver: Solver
    ) -> Solution: ...

class StochasticModel:
    """Stochastic constitutive RMA expression model."""

    def __init__(
        self,
        prod: float = 0.2,
        bbb_transport: float = 0.6,
        deg: float = 0.007,
        prod_noise: float = 0.5,
        seed: int = 42,
    ) -> None: ...
    def solve(
        self, t0: float, tf: float, dt: float, init_state: State, solver: Solver
    ) -> Solution: ...

class State:
    """Constitutive model state."""

    def __init__(self, brain_rma: float = 0.0, plasma_rma: float = 0.0) -> None: ...
    @property
    def brain_rma(self) -> float: ...
    @brain_rma.setter
    def brain_rma(self, value: float) -> None: ...
    @property
    def plasma_rma(self) -> float: ...
    @plasma_rma.setter
    def plasma_rma(self, value: float) -> None: ...
