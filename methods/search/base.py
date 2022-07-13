from abc import ABC, abstractmethod
from typing import List, Optional

from data_collection.utils import TOKENS_KEY
from methods import DataOut
from preprocess import PreProcessor


class BaseSearcher(ABC):
    def __init__(self, data, tokens_key: str = TOKENS_KEY):
        self.data = data
        self.tokens_key = tokens_key
        self.pre_processor = PreProcessor()

    @abstractmethod
    def process_query(self, query):
        pass

    @abstractmethod
    def search(self, query, k: int = 10) -> Optional[DataOut]:
        pass
