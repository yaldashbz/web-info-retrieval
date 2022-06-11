from abc import abstractmethod, ABC

from sklearn.metrics import silhouette_score

from methods.utils import plot_silhouette


class BaseValidator(ABC):
    def __init__(self, use_plot: bool = False):
        self.use_plot = use_plot

    @abstractmethod
    def plot(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def validate(self, results, df):
        raise NotImplementedError


class SilhouetteValidator(BaseValidator):
    def plot(self, df, n, labels, avg):
        plot_silhouette(df, n, labels, avg)

    def validate(self, results, df):
        df = df.to_numpy()
        avg_dict = dict()
        for n, cluster in results.items():
            labels = cluster.predict(df)
            avg = silhouette_score(df, labels)
            avg_dict[avg] = n
            if self.use_plot:
                self.plot(df, n, labels, avg)
