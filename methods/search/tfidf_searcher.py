import re
import numpy as np
from itertools import chain
from methods import TFIDFRepresentation, cosine_sim, TOKENS_KEY
from methods.search.base import BaseSearcher


class TFIDFSearcher(BaseSearcher):
    def __init__(self, data, tokens_key: str = TOKENS_KEY):
        super().__init__(data, tokens_key)
        self.representation = TFIDFRepresentation(data, tokens_key=tokens_key)

    def process_query(self, query):
        query = re.sub('\\W+', ' ', query).strip()
        return self.pre_processor.process(query)

    def search(self, query, k):
        scores = list()
        tokens = self.process_query(query)
        query_vector = self._get_query_vector(tokens)
        for doc in self.representation.matrix.A:
            scores.append(cosine_sim(query_vector, doc))

        return self._get_results(scores, k)

    def _get_results(self, scores, k):
        out = np.array(scores).argsort()[-k:][::-1]
        return [dict(
            index=index,
            url=self.data[index]['url'],
            score=scores[index]
        ) for index in out]

    def _get_query_vector(self, tokens):
        n = len(self.representation.vocab)
        vector = np.zeros(n)

        for token in chain(*tokens):
            try:
                index = self.representation.tfidf.vocabulary_[token]
                vector[index] = 1
            except ValueError:
                pass
        return vector
