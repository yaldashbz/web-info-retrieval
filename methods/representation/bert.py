import os
import torch
import pandas as pd
from typing import List
from sentence_transformers import SentenceTransformer

from data_collection.utils import get_contents, TOKENS_KEY
from methods.representation.base import BaseRepresentation


class BertRepresentation(BaseRepresentation):
    _PATH = '../embeddings'
    _FILE = 'bert_embeddings.json'

    def __init__(self, data, load: bool = False, tokens_key: str = TOKENS_KEY):
        super().__init__(data, tokens_key)
        contents = self.prepare_data()

        if load and not os.path.exists(os.path.join(self._PATH, self._FILE)):
            raise ValueError

        self.model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
        self._to_cuda()

        if not load:
            self.df = self._get_embeddings(contents)
            self._save_embeddings()
        else:
            self.df = self._load_embeddings()
        self.embeddings = self.df.values.tolist()

    def _load_embeddings(self):
        return pd.read_json(os.path.join(self._PATH, self._FILE))

    def _save_embeddings(self):
        if not os.path.exists(self._PATH):
            os.mkdir(self._PATH)
        self.df.to_json(os.path.join(self._PATH, self._FILE))

    def prepare_data(self) -> List:
        return get_contents(self.data, key=self.tokens_key)

    def represent(self) -> pd.DataFrame:
        return self.df

    def _to_cuda(self):
        if torch.cuda.is_available():
            self.model = self.model.to(torch.device('cuda'))

    def _get_embeddings(self, contents):
        embeddings = self.model.encode(
            contents,
            show_progress_bar=True,
            normalize_embeddings=True
        )
        return pd.DataFrame(embeddings)
