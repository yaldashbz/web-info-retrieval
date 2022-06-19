from typing import Dict

import numpy as np
import networkx as nx

from methods.linking.graph import GraphBuilder


class ContentLinkAnalyser:
    def __init__(self, data, weighted: bool = True, **builder_kwargs):
        self.builder = GraphBuilder(data=data, **builder_kwargs)
        self.graph = self.builder.build_weighted() if weighted else self.builder.build()
        self.pagerank = None
        self.hubs = None
        self.authorities = None

    def _get_most_relevant(self, ranking: Dict[int, float]):
        return self.builder.paragraphs[np.argmax(list(ranking.values()))]

    def apply_pagerank(self):
        self.pagerank = nx.pagerank(self.graph)
        return self._get_most_relevant(self.pagerank)

    def apply_hits(self):
        self.hubs, self.authorities = nx.hits(self.graph)
        return (
            self._get_most_relevant(self.hubs),
            self._get_most_relevant(self.authorities)
        )


if __name__ == '__main__':
    a = ContentLinkAnalyser(
        data=[[['my', 'name', 'is', 'yalda'], ['salaam', 'chetori'], ['this', 'is', 'test', 'yalda']],
              [['my', 'ghodghs', 'is', 'yalda'], ['erf', 'writes'], ['am', 'is', 'pl']]],
        sent_num=2, min_similar=2,
        weighted=True)
    p = a.apply_pagerank()
    print(p, a.pagerank)