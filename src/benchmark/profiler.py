import time
import statistics
import numpy as np
from typing import Dict, Callable
from ..algorithms.base_search import StringSearchAlgorithm

class SearchProfiler:
    @staticmethod
    def benchmark(
        algorithm: StringSearchAlgorithm,
        messages: list[str],
        pattern: str,
        repetitions: int
    ) -> Dict[str, float]:
        times = []
        
        for _ in range(repetitions):
            start = time.perf_counter()
            for msg in messages:
                algorithm.search(msg, pattern)
            times.append(time.perf_counter() - start)

        per_msg = [t / len(messages) * 1e3 for t in times]
        return {
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
