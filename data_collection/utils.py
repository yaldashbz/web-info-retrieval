from itertools import chain
from typing import List, Tuple
from nltk import FreqDist

DIVIDER = ' '

CATEGORIES = [
    'religion',
    'sports', 'drink',
    'financial', 'health', 'literature',
    'social networks', 'food', 'history',
    'animals', 'news', 'science', 'movies',
    'music', 'games', 'computer',
    'football', 'basketball', 'volleyball',
    'university', 'national', 'politics'
]


def get_keywords(doc: List[List[str]], count: int = 20) -> List[Tuple]:
    words = get_words(doc)
    return FreqDist(words).most_common(count)


def get_contents(data: List):
    return [
        DIVIDER.join([DIVIDER.join(sentence) for sentence in doc['tokens']])
        for doc in data
    ]


def get_sentences(doc):
    return [' '.join(words) for words in doc['tokens']]


def get_words(doc):
    return list(chain(*doc['tokens']))
