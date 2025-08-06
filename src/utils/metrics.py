# src/utils/metrics.py
from typing import List, Tuple

def compute_confusion_matrix(
    true_labels: List[str],
    pred_labels: List[str]
) -> Tuple[int, int, int, int]:
    TP = sum(1 for t,p in zip(true_labels,pred_labels) if t=="spam" and p=="spam")
    FP = sum(1 for t,p in zip(true_labels,pred_labels) if t=="ham"  and p=="spam")
    TN = sum(1 for t,p in zip(true_labels,pred_labels) if t=="ham"  and p=="ham")
    FN = sum(1 for t,p in zip(true_labels,pred_labels) if t=="spam" and p=="ham")
    return TP, FP, TN, FN

def precision(TP: int, FP: int) -> float:
    return TP / (TP + FP) if (TP + FP) > 0 else 0.0

def recall(TP: int, FN: int) -> float:
    return TP / (TP + FN) if (TP + FN) > 0 else 0.0

def f1_score(prec: float, rec: float) -> float:
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
