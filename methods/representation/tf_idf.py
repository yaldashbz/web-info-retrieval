import os
import pickle
import re
from itertools import chain
from typing import List

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from data_collection.utils import get_contents, TOKENS_KEY
from methods.representation.base import BaseRepresentation
from preprocess import PreProcessor


class TFIDFRepresentation(BaseRepresentation):
    _PATH = '../models/tfidf'
    _FILE = 'vectorizer.pkl'

    def __init__(self, data, tokens_key: str = TOKENS_KEY, load: bool = False):
        super().__init__(data, tokens_key)
        self.tfidf = self._get_tfidf(load)

        if data and not load:
            contents = self.prepare_data()
            self.matrix = self.tfidf.fit_transform(contents)
            self.save_model(self.file_path)

        self.vocab = self.tfidf.get_feature_names_out()

    @property
    def file_path(self):
        return os.path.join(self._PATH, self._FILE)

    def _get_tfidf(self, load: bool):
        return TfidfVectorizer(use_idf=True, norm='l2', analyzer='word') \
            if not load else self.load_model(self.file_path)

    def save_model(self, path: str):
        pickle.dump(self.tfidf, open(path, 'wb'))

    @classmethod
    def load_model(cls, path: str):
        return pickle.load(open(path, 'rb'))

    def prepare_data(self) -> List:
        return get_contents(self.data, key=self.tokens_key)

    def represent(self) -> pd.DataFrame:
        return pd.DataFrame(data=self.matrix.toarray(), columns=self.vocab)

    @classmethod
    def process_query(cls, query: str):
        preprocessor = PreProcessor()
        query = re.sub('\\W+', ' ', query).strip()
        return preprocessor.process(query)

    def embed(self, query: str):
        tokens = self.process_query(query)
        n = len(self.vocab)
        vector = np.zeros(n)
        for token in chain(*tokens):
            try:
                index = self.tfidf.vocabulary_[token]
                vector[index] = 1
            except ValueError:
                pass
        return vector
