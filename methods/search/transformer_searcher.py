import itertools
import numpy as np

from methods.utils import cosine_sim
from methods.search.base import BaseSearcher
from methods.representation import BertRepresentation
from data_collection.utils import TOKENS_KEY


class TransformerSearcher(BaseSearcher):
    def __init__(self, data, load: bool = False, tokens_key: str = TOKENS_KEY):
        super().__init__(data, tokens_key)
        self.representation = BertRepresentation(data=data, load=load, tokens_key=tokens_key)

    def process_query(self, query):
        tokens = self.pre_processor.process(query)
        tokens = itertools.chain(*tokens)
        query = ' '.join(tokens)
        return [query]

    def _search(self, query, k):
        query = self.process_query(query)
        query_embedding = self.representation.model.encode(
            query, show_progress_bar=True, normalize_embeddings=True
        )
        similarities = self._get_similarities(query_embedding, k)
        return [dict(url=url, score=score) for url, score in similarities]

    def _get_similarities(self, query_embedding, k):
        similarities = dict()
        for url, embedding in self.representation.embeddings.items():
            similarities[url] = cosine_sim(
                np.mean(embedding, axis=0).reshape(1, 768),
                query_embedding.reshape(768, 1)
            )
        return sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:k]
