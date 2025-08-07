import random
from typing import Dict, List

class PatternGenerator:
    @staticmethod
    def generate_patterns(
        messages: List[str],
        sizes: List[int],
        samples_per_size: int,
        seed: int
    ) -> Dict[int, List[str]]:
        random.seed(seed)
        patterns = {}
        for size in sizes:
            pool = [msg for msg in messages if len(msg) >= size]
            patterns[size] = []
            for _ in range(samples_per_size):
                msg = random.choice(pool)
                start = random.randint(0, len(msg) - size)
                patterns[size].append(msg[start:start+size])
        return patterns
