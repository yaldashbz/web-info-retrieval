from typing import Tuple

import pandas as pd
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer

from data_collection.utils import get_content
from methods.representation import TFIDFRepresentation, BertRepresentation, FasttextRepresentation

_representations = {
    'tf-idf': TFIDFRepresentation,
    'bert': BertRepresentation,
    'fasttext': FasttextRepresentation
}


class ContentKMeanCluster:
    def __init__(self, data, method: str = 'tf-idf', **repr_kwargs):
        self.data = data
        self.representation = _representations[method](data=data, **repr_kwargs)
        self.represented_df = self.representation.represent()
        self.k_means = None
        self.estimator = None

    def run(self, k: int = 2):
        self.k_means = KMeans(n_clusters=k, random_state=1)
        self.estimator = self.k_means.fit(self.represented_df)
        return self.k_means, self.estimator

    def _get_result(self):
        result = self.data
        for doc_id, cluster_id in enumerate(self.k_means.labels_):
            result[doc_id].update(cluster_id=cluster_id)
        return result

    def analyse(self) -> pd.DataFrame:
        assert self.k_means and self.estimator

        result = self._get_result()
        result_df = pd.DataFrame(result)
        result_df['content'] = result_df['tokens'].apply(get_content)
        result_df.pop('tokens')
        return result_df

    def elbow_visualize(self, k_range: Tuple[int, int]):
        model = KMeans(random_state=1)
        visualizer = KElbowVisualizer(model, k=k_range).fit(self.represented_df)
        visualizer.show()
