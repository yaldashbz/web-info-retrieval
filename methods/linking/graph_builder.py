import networkx as nx
import matplotlib.pyplot as plt
from itertools import chain
from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from tqdm import tqdm

from data_collection.utils import get_sentences
from methods.linking.utils import count_same_words


class GraphBuilder:
    def __init__(
            self,
            dataset: List,
            sent_num: int,
            min_similar: float
    ):
        self.graph = nx.Graph()
        self.sent_num = sent_num
        self.min_similar = max(0, min_similar)
        self.paragraphs = self._get_paragraphs(dataset)

    def _get_paragraphs(self, data):
        num = self.sent_num
        return [list(chain(*data[i:i + num]))
                for i in range(0, len(data), num)]

    def _get_nodes(self):
        return list(range(len(self.paragraphs)))

    def _get_score(self, node, other):
        first = self.paragraphs[node]
        second = self.paragraphs[other]
        return count_same_words(first, second)

    def _has_edge(self, node, other):
        score = self._get_score(node, other)
        return score >= self.min_similar

    def _get_edges(self, nodes):
        return [(node, other) for other in nodes
                for node in tqdm(nodes) if self._has_edge(node, other) and node != other]

    def _get_weighted_edges(self, nodes):
        edges = []
        for node in tqdm(nodes):
            for other in nodes:
                if node != other:
                    score = self._get_score(node, other)
                    if score >= self.min_similar:
                        edges.append((node, other, score))
        return edges

    def build(self):
        self.graph = nx.Graph()
        nodes = self._get_nodes()
        edges = self._get_edges(nodes)
        self.graph.add_nodes_from(nodes)
        self.graph.add_edges_from(edges)
        return self.graph

    def build_weighted(self):
        self.graph = nx.Graph()
        nodes = self._get_nodes()
        edges = self._get_weighted_edges(nodes)
        self.graph.add_nodes_from(nodes)
        self.graph.add_weighted_edges_from(edges)
        return self.graph

    def show(self, **kwargs):
        pos = nx.spiral_layout(self.graph)
        nx.draw(self.graph, pos, with_labels=True, **kwargs)
        plt.show()


class TFIDFGraphBuilder(GraphBuilder):
    def __init__(
            self,
            dataset: List,
            sent_num: int,
            min_similar: float
    ):
        super().__init__(dataset, sent_num, min_similar)
        self.tfidf = TfidfVectorizer(use_idf=True, norm='l2', analyzer='word')
        contents = get_sentences(self.paragraphs)
        self.matrix = self.tfidf.fit_transform(contents)

    def build(self):
        return self.build_weighted()

    def build_weighted(self):
        P = self.matrix.dot(self.matrix.T)
        P_norm = normalize(P, norm='l1')
        self.graph = nx.from_numpy_array(P_norm)
        return self.graph
