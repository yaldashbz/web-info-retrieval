import itertools
from typing import Optional

import faiss
import numpy as np

from methods.search.base import BaseSearcher
from methods.search.utils import DataOut
from methods.representation import BertRepresentation
from data_collection.utils import TOKENS_KEY


class TransformerSearcher(BaseSearcher):
    def __init__(
            self, data,
            load: bool = False,
            tokens_key: str = TOKENS_KEY
    ):
        super().__init__(data, tokens_key)
        self.representation = BertRepresentation(
            data=data, load=load, tokens_key=tokens_key
        )
        self.index = self._get_index(self.representation.embeddings)

    def _get_index(self, embeddings):
        index = faiss.IndexIDMap(faiss.IndexFlatL2(embeddings.shape[1]))
        index.add_with_ids(embeddings, np.array(range(len(self.data))))
        return index

    def process_query(self, query):
        tokens = self.pre_processor.process(query)
        tokens = itertools.chain(*tokens)
        query = ' '.join(tokens)
        return [query]

    def search(self, query, k: int = 10) -> Optional[DataOut]:
        query = self.process_query(query)
        vector = self.representation.model.encode(
            query, show_progress_bar=True, normalize_embeddings=True
        )
        distances, indexes = self.index.search(np.array(vector).astype('float32'), k=k)
        return DataOut(self._get_results(distances, indexes))

    def _get_results(self, distances, indexes):
        indexes = indexes.flatten().tolist()
        distances = distances.flatten().tolist()
        return [dict(
            url=self.data[index]['url'],
            distance=distances[i]
        ) for i, index in enumerate(indexes)]
