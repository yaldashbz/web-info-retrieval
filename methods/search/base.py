from abc import ABC, abstractmethod
from typing import List

from data_collection.utils import TOKENS_KEY
from methods.search.utils import DataOut
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
    def _search(self) -> List:
        raise NotImplementedError

    def search(self, query, k):
        return DataOut(self._search())
