import itertools
import numpy as np

from methods import cosine_sim, FasttextRepresentation
from methods.search.base import BaseSearcher


class FasttextSearcher(BaseSearcher):
    def __init__(self, data, train: bool = True, min_count: int = 1):
        super().__init__(data)
        self.representation = FasttextRepresentation(data, train=train, min_count=min_count)

    def _get_query_embedding_avg(self, tokens):
        return np.mean([self.representation.fasttext.wv[token] for token in tokens], axis=0)

    def process_query(self, query):
        tokens = self.pre_processor.process(query)
        return list(itertools.chain(*tokens))

    def search(self, query, k):
        tokens = self.process_query(query)
        query_embedding_avg = self._get_query_embedding_avg(tokens)
        similarities = self._get_similarities(query_embedding_avg, k)
        return self._get_result(similarities)

    def _get_similarities(self, query_embedding, k):
        similarities = dict()
        for index, embedding in self.representation.doc_embedding_avg.items():
            similarities[index] = cosine_sim(embedding, query_embedding)
        return sorted(similarities.items(), key=lambda x: x[1])[::-1][:k]

    def _get_result(self, similarities):
        return [dict(
            url=self.data[index]['url'],
            score=score
        ) for index, score in similarities]
