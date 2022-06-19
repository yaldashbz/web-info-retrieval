# TODO: remove

from abc import abstractmethod, ABC

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from methods.utils import plot_silhouette


class BaseValidator(ABC):
    @abstractmethod
    def plot(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def validate(self, **kwargs):
        raise NotImplementedError


class SilhouetteValidator(BaseValidator):
    def plot(self, df, n, labels, score):
        plot_silhouette(df, n, labels, score)

    def validate(self, estimator: KMeans, k, repr_df, use_plot: bool = True):
        df = repr_df.to_numpy()
        labels = estimator.predict(df)
        score = silhouette_score(df, labels)
        if use_plot:
            self.plot(df, k, labels, score)
        return score


class RSSValidator(BaseValidator):
    def plot(self, **kwargs):
        return None

    def validate(self, estimator: KMeans):
        return estimator.inertia_
