from __future__ import annotations

import time
from types import TracebackType

class Time:
    name: str
    start_time: float
    stop_time: float
    
    def __init__(self, name: str):
        self.name = name
    
    def __enter__(self) -> Time:
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(
        self,
        type_: type[BaseException] | None,
        value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        self.stop_time = time.perf_counter()
        print(f"{self.name} time: {self.stop_time - self.start_time:.3g} s")
    