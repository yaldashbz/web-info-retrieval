from typing import Dict, Any

from sklearn.cluster import KMeans

from methods.representation import TFIDFRepresentation
from methods.validator import SilhouetteValidator


class ContentKMeanCluster:
    def __init__(self, data):
        self.data = data
        self.representation = TFIDFRepresentation()

    @classmethod
    def _get_results(cls, data, max_k: int = 2) -> Dict[int, Any]:
        max_k += 1
        results = dict()
        for k in range(2, max_k):
            kmeans = KMeans(
                n_clusters=k,
                random_state=1,
                algorithm='full'
            )
            results[k] = kmeans.fit(data)
        return results

    def run(self, max_k: int = 2):
        df = self.representation.represent(self.data)
        return self._get_results(df, max_k), df

    def silhouette_validate(self, max_k: int):
        validator = SilhouetteValidator(use_plot=True)
        results, df = self.run(max_k)
        validator.validate(results, df)
