import torch
import pandas as pd
from typing import List
from sentence_transformers import SentenceTransformer

from data_collection.utils import get_contents
from methods.representation.base import BaseRepresentation


class BertRepresentation(BaseRepresentation):
    def __init__(self, data):
        super().__init__(data)
        contents = self.prepare_data()
        self.model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
        self._to_cuda()
        self.df = self._get_embeddings(contents)

    def prepare_data(self) -> List:
        return get_contents(self.data)

    def represent(self) -> pd.DataFrame:
        return self.df

    def _to_cuda(self):
        if torch.cuda.is_available():
            self.model = self.model.to(torch.device('cuda'))

    def _get_embeddings(self, contents):
        embeddings = self.model.encode(contents, show_progress_bar=True, normalize_embeddings=True)
        return pd.DataFrame(embeddings)
