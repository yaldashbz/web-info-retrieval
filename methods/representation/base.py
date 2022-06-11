import pandas as pd
from typing import List
from abc import abstractmethod, ABC


class BaseRepresentation(ABC):
    def __init__(self, data):
        self.data = data

    def prepare_data(self) -> List:
        return self.data

    @abstractmethod
    def represent(self) -> pd.DataFrame:
        raise NotImplementedError
