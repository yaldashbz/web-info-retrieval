from sklearn.cluster import KMeans

from methods.representation import TFIDFRepresentation, BertRepresentation, FasttextRepresentation

_representations = {
    'tf-idf': TFIDFRepresentation,
    'bert': BertRepresentation,
    'fasttext': FasttextRepresentation
}


class ContentKMeanCluster:
    def __init__(self, data, method: str = 'tf-idf'):
        self.representation = _representations[method](data)
        self.represented_df = self.representation.represent()

    def run(self, k: int = 2):
        kmeans = KMeans(n_clusters=k, random_state=1)
        estimator = kmeans.fit(self.represented_df)
        return kmeans, estimator
