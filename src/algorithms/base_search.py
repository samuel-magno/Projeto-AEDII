from abc import ABC, abstractmethod

class StringSearchAlgorithm(ABC):
    @abstractmethod
    def search(self, text: str, pattern: str) -> int:
        pass