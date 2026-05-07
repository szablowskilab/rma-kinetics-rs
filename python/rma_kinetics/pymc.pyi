from typing import Any

from pytensor.graph.op import Op

class InferenceOp(Op):
    solver: Any
    def __init__(self, solver: Any) -> None: ...

class InferenceVJPOp(Op):
    solver: Any
    def __init__(self, solver: Any) -> None: ...

class PopulationInferenceOp(Op):
    solver: Any
    def __init__(self, solver: Any) -> None: ...

class PopulationInferenceVJPOp(Op):
    solver: Any
    def __init__(self, solver: Any) -> None: ...

__all__: list[str]
