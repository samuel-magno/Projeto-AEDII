# src/classifier/spam_classifier.py
from typing import List
from src.algorithms.base_search import StringSearchAlgorithm
from src.algorithms.boyer_moore import BoyerMooreSearch

class SpamClassifier:
    def __init__(
        self,
        patterns: List[str],
        algorithm: StringSearchAlgorithm,
        threshold: int = 1
    ):
        self.patterns = [p.lower() for p in patterns]
        self.algorithm = algorithm
        self.threshold = threshold

        if isinstance(self.algorithm, BoyerMooreSearch):
            for pat in self.patterns:
                self.algorithm.preprocess(pat)

    def classify(self, text: str) -> str:
        count = 0
        lower_text = text.lower()

        for pat in self.patterns:
            if self.algorithm.search(lower_text, pat) != -1:
                count += 1
                if count >= self.threshold:
                    return "spam"

        return "ham"

