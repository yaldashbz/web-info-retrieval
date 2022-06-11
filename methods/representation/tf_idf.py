import os
import pandas as pd
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer

from data_collection.utils import get_contents
from methods.representation.base import BaseRepresentation


class TFIDFRepresentation(BaseRepresentation):
    _PATH = 'matrices'
    _FILE = 'tfidf.json'

    def __init__(self, data, load: bool = False):
        super().__init__(data)
        contents = self.prepare_data()
        self.tfidf = TfidfVectorizer(use_idf=True, norm='l2', analyzer='word')
        if not load:
            self.matrix = self.tfidf.fit_transform(contents)
            self.vocab = self.tfidf.get_feature_names_out()
            self.df = pd.DataFrame(data=self.matrix.toarray(), columns=self.vocab)
            self._save()
        else:
            self.df = self._load()
            self.matrix = self.df.to_numpy()
            self.vocab = list(self.df.columns)

    def _load(self):
        return pd.read_json(os.path.join(self._PATH, self._FILE))

    def _save(self):
        if not os.path.exists(self._PATH):
            os.mkdir(self._PATH)
        self.df.to_json(os.path.join(self._PATH, self._FILE))

    def prepare_data(self) -> List:
        return get_contents(self.data)

    def represent(self) -> pd.DataFrame:
        return self.df
