from typing import List


def count_same_words(first: List[str], second: List[str]):
    return len(set(first).intersection(set(second)))
