from itertools import chain
from typing import Dict

import numpy as np
import networkx as nx

from methods.linking.graph import GraphBuilder


class ContentLinkAnalyser:
    def __init__(
            self,
            dataset,
            cleaned_dataset,
            weighted: bool = True,
            sent_num: int = 3,
            min_similar: int = 5
    ):
        self.dataset = dataset
        self.cleaned_dataset = cleaned_dataset
        self.sent_num = sent_num
        self.builder = GraphBuilder(
            dataset=list(chain(*dataset)),
            sent_num=sent_num,
            min_similar=min_similar
        )
        self.graph = self.builder.build_weighted() if weighted else self.builder.build()
        self.pagerank = None
        self.hubs = None
        self.authorities = None

    def _get_most_relevant(self, rank: int):
        return ' '.join(list(chain(*list(chain(
            *self.dataset))[rank * self.sent_num: rank * self.sent_num + self.sent_num])))

    def _get_most_cleaned_relevant(self, rank: int):
        return self.builder.paragraphs[rank]

    def apply_pagerank(self):
        self.pagerank = nx.pagerank(self.graph)
        rank = np.argmax(list(self.pagerank.values()))
        return self._get_most_relevant(rank), self._get_most_cleaned_relevant(rank)

    def apply_hits(self):
        self.hubs, self.authorities = nx.hits(self.graph)
        h_rank = np.argmax(list(self.hubs.values()))
        a_rank = np.argmax(list(self.authorities.values()))
        return (
            self._get_most_relevant(h_rank),
            self._get_most_cleaned_relevant(h_rank),
            self._get_most_relevant(a_rank),
            self._get_most_cleaned_relevant(a_rank)
        )
