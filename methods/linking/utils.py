from typing import List


def count_same_words(first: List[str], second: List[str]):
    return len([w for w in first if w in second])
