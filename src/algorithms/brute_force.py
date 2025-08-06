from .base_search import StringSearchAlgorithm

class BruteForceSearch(StringSearchAlgorithm):
    def search(self, text: str, pattern: str) -> int:
        n, m = len(text), len(pattern)
        for i in range(n - m + 1):
            if text[i:i+m] == pattern:
                return i
        return -1
