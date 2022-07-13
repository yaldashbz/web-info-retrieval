from abc import ABC, abstractmethod

from preprocess import PreProcessor


class BaseSearcher(ABC):
    def __init__(self, data):
        self.data = data
        self.pre_processor = PreProcessor()

    @abstractmethod
    def process_query(self, query):
        pass

    @abstractmethod
    def search(self, query, k):
        pass
