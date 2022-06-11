from sklearn.cluster import KMeans

from methods.representation import TFIDFRepresentation


class ContentKMeanCluster:
    def __init__(self, data):
        self.data = data
        self.representation = TFIDFRepresentation()
        self.represented_df = self.representation.represent(self.data)

    @classmethod
    def _kmeans_fit(cls, data, k: int = 2):
        kmeans = KMeans(
            n_clusters=k,
            random_state=1,
            algorithm='full'
        )
        return kmeans.fit(data)

    def run(self, k: int = 2):
        df = self.represented_df
        return dict(estimator=self._kmeans_fit(df, k), k=k)

