import time
import statistics
import numpy as np
import sys
from typing import Dict, Any
from ..algorithms.base_search import StringSearchAlgorithm
from ..classifier.spam_classifier import SpamClassifier

def _deep_getsizeof(obj, seen=None):
    if seen is None:
        seen = set()
    oid = id(obj)
    if oid in seen:
        return 0
    seen.add(oid)
    size = sys.getsizeof(obj)
    if isinstance(obj, dict):
        for k, v in obj.items():
            size += _deep_getsizeof(k, seen) + _deep_getsizeof(v, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        for item in obj:
            size += _deep_getsizeof(item, seen)
    return size

class SearchProfiler:
    @staticmethod
    def benchmark(
        algorithm: StringSearchAlgorithm,
        messages: list[str],
        pattern: str,
        repetitions: int
    ) -> Dict[str, float]:
        if hasattr(algorithm, 'preprocess'):
            algorithm.preprocess(pattern)
            space_bytes = _deep_getsizeof(getattr(algorithm, 'bad_char_table', {}))
        else:
            space_bytes = 0

        times = []
        for _ in range(repetitions):
            start = time.perf_counter()
            for msg in messages:
                algorithm.search(msg, pattern)
            times.append(time.perf_counter() - start)

        per_msg = [t / len(messages) * 1e3 for t in times]
        stats = {
            "mean_s": statistics.mean(times),
            "stdev_s": statistics.stdev(times) if repetitions > 1 else 0.0,
            "min_s": min(times),
            "max_s": max(times),
            "p90_s": float(np.percentile(times, 90)),
            "p95_s": float(np.percentile(times, 95)),
            "mean_per_msg_ms": statistics.mean(per_msg),
            "p90_per_msg_ms": float(np.percentile(per_msg, 90)),
            "p95_per_msg_ms": float(np.percentile(per_msg, 95))
        }
        stats["space_bytes"] = space_bytes
        return stats

    @staticmethod
    def benchmark_classification(
        classifier: SpamClassifier,
        texts: list[str],
        repetitions: int
    ) -> Dict[str, Any]:
        space = _deep_getsizeof(classifier.patterns)
        alg = getattr(classifier, 'algorithm', None)
        if alg and hasattr(alg, 'bad_char_table'):
            space += _deep_getsizeof(getattr(alg, 'bad_char_table', {}))

        times = []
        for _ in range(repetitions):
            start = time.perf_counter()
            for t in texts:
                classifier.classify(t)
            times.append(time.perf_counter() - start)

        per_msg = [t / len(texts) * 1e3 for t in times]
        stats = {
            "mean_s": statistics.mean(times),
            "stdev_s": statistics.stdev(times) if repetitions > 1 else 0.0,
            "min_s":   min(times),
            "max_s":   max(times),
            "p90_s": float(np.percentile(times, 90)),
            "p95_s": float(np.percentile(times, 95)),
            "mean_per_msg_ms":  statistics.mean(per_msg),
            "p90_per_msg_ms":   float(np.percentile(per_msg, 90)),
            "p95_per_msg_ms":   float(np.percentile(per_msg, 95)),
            "space_bytes": space
        }
        return stats
