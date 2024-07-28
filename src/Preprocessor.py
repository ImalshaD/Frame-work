from abc import ABC, abstractmethod
from __future__ import annotations

class Preprocessor(ABC):
    def __init__(self, chain : Preprocessor = None) -> None:
        super().__init__()
        self._next : Preprocessor = chain
    
    def setNext(self, chain : Preprocessor):
        self._next = chain

    