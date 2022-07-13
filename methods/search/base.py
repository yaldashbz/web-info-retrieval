from abc import ABC, abstractmethod

from methods.search import TOKENS_KEY
from preprocess import PreProcessor


class BaseSearcher(ABC):
    def __init__(self, data, tokens_key: str = TOKENS_KEY):
        self.data = data
        self.tokens_key = tokens_key
        self.pre_processor = PreProcessor()

    @abstractmethod
    def process_query(self, query):
        raise NotImplementedError

    @abstractmethod
    def search(self, query, k):
        raise NotImplementedError
