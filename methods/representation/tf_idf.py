import os
import pandas as pd
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer

from data_collection.utils import get_contents
from methods.representation.base import BaseRepresentation


class TFIDFRepresentation(BaseRepresentation):
    # TODO: remove load
    def __init__(self, data, load: bool = False):
        super().__init__(data)
        contents = self.prepare_data()

        self.tfidf = TfidfVectorizer(use_idf=True, norm='l2', analyzer='word')
        self.matrix = self.tfidf.fit_transform(contents)
        self.vocab = self.tfidf.get_feature_names_out()

    def prepare_data(self) -> List:
        return get_contents(self.data)

    def represent(self) -> pd.DataFrame:
        return pd.DataFrame(data=self.matrix.toarray(), columns=self.vocab)
