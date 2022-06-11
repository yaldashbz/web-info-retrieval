from typing import List

from abc import abstractmethod, ABC

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from data_collection.utils import get_contents


class BaseRepresentation(ABC):
    @classmethod
    def prepare_data(cls, data) -> List:
        return data

    @abstractmethod
    def represent(self, data) -> pd.DataFrame:
        raise NotImplementedError


class TFIDFRepresentation(BaseRepresentation):
    @classmethod
    def prepare_data(cls, data) -> List:
        return get_contents(data)

    def represent(self, data) -> pd.DataFrame:
        contents = self.prepare_data(data)
        tfidf = TfidfVectorizer(use_idf=True, norm='l2', analyzer='word')
        matrix = tfidf.fit_transform(contents)
        vocab = tfidf.get_feature_names_out()
        return pd.DataFrame(data=matrix.toarray(), columns=vocab)
