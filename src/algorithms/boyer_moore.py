from .base_search import StringSearchAlgorithm

class BoyerMooreSearch(StringSearchAlgorithm):
    def __init__(self):
        self.bad_char_table = None

    def preprocess(self, pattern: str):
        tbl = [-1] * 256
        for i, ch in enumerate(pattern):
            tbl[ord(ch)] = i
        self.bad_char_table = tbl

    def search(self, text: str, pattern: str) -> int:
        if not self.bad_char_table:
            self.preprocess(pattern)
            
        n, m = len(text), len(pattern)
        s = 0
        while s <= n - m:
            j = m - 1
            while j >= 0 and pattern[j] == text[s+j]:
                j -= 1
            if j < 0:
                return s
            bc = ord(text[s+j])
            s += max(1, j - self.bad_char_table[bc] if bc < 256 else 1)
        return -1
