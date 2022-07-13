import pandas as pd
from typing import List
from abc import abstractmethod, ABC

from methods.search import TOKENS_KEY


class BaseRepresentation(ABC):
    def __init__(self, data, tokens_key: str = TOKENS_KEY):
        self.data = data
        self.tokens_key = tokens_key

    def prepare_data(self) -> List:
        return self.data

    @abstractmethod
    def represent(self) -> pd.DataFrame:
        raise NotImplementedError
