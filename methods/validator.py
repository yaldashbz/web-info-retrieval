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
    def validate(self, **kwargs):
        raise NotImplementedError


class SilhouetteValidator(BaseValidator):
    def plot(self, df, n, labels, avg):
        plot_silhouette(df, n, labels, avg)

    def validate(self, estimator, k, df):
        df = df.to_numpy()
        avg_dict = dict()
        labels = estimator.predict(df)
        avg = silhouette_score(df, labels)
        avg_dict[avg] = k
        if self.use_plot:
            self.plot(df, k, labels, avg)
