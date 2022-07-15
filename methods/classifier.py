import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from methods import (
    TFIDFRepresentation,
    BertRepresentation,
    FasttextRepresentation
)

_representations = {
    'tf-idf': TFIDFRepresentation,
    'bert': BertRepresentation,
    'fasttext': FasttextRepresentation
}


class _BaseClassifier:
    def __init__(
            self, data,
            method: str = 'tf-idf',
            split_test_size: float = 0.1,
            split_random_state: float = 1,
            **repr_kwargs
    ):
        self.data = data
        self.representation = _representations[method](data=data, **repr_kwargs)
        self.X, self.y = self._getXy()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y,
            test_size=split_test_size,
            random_state=split_random_state
        )
        self.y_predicted = None

    def _getXy(self):
        X = self.representation.df.values()
        y = np.array([
            doc['category'] if doc['category'] else 'others'
            for doc in self.data
        ])
        return X, y

    def f1_score(self):
        f1_score(self.y_test, self.y_predicted, average='macro')


class LogisticRegressionClassifier(_BaseClassifier):
    def __call__(self, random_state: float = 0):
        classifier = LogisticRegression(random_state=random_state).fit(
            self.X_train, self.y_train
        )
        self.y_predicted = classifier.predict(self.X_test)
        return classifier


class TransformerClassifier(_BaseClassifier):
    def classify(self):
        pass
