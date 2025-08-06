# src/data/vocab_builder.py
from collections import Counter
from typing import List, Tuple

def build_spam_vocab(labels: List[str], texts: List[str], k: int) -> List[str]:
    spam_counts = Counter()
    ham_counts = Counter()
    for label, text in zip(labels, texts):
        tokens = text.split()
        if label == "spam":
            spam_counts.update(tokens)
        else:
            ham_counts.update(tokens)

    scores: List[Tuple[str, float]] = []
    for token, s_count in spam_counts.items():
        h_count = ham_counts.get(token, 0)
        ratio = (s_count + 1) / (h_count + 1)
        scores.append((token, ratio))

    top_k = sorted(scores, key=lambda x: x[1], reverse=True)[:k]
    return [token for token, _ in top_k]
