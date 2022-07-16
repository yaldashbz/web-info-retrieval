import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, plot_confusion_matrix, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

from data_collection import CATEGORIES
from data_collection.utils import OTHERS
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
        self.classifier = None

    def _getXy(self):
        X = self.representation.represent().values
        y = np.array([
            doc['category'] if doc['category'] else OTHERS
            for doc in self.data
        ])
        return X, y

    def f1_score(self):
        return f1_score(self.y_test, self.y_predicted, average='macro')

    def accuracy(self):
        return accuracy_score(self.y_test, self.y_predicted)

    def confusion_matrix(self, plot: bool = True):
        if plot:
            plot_confusion_matrix(self.classifier, self.X_test, self.y_test)
        return confusion_matrix(self.y_test, self.y_predicted, labels=CATEGORIES + [OTHERS])


class LogisticRegressionClassifier(_BaseClassifier):
    def classify(self, random_state: float = 0):
        self.classifier = LogisticRegression(
            random_state=random_state,
            max_iter=300
        ).fit(
            self.X_train, self.y_train
        )
        self.y_predicted = self.classifier.predict(self.X_test)
        return self.classifier


class NaiveBayesClassifier(_BaseClassifier):
    def classify(self):
        self.classifier = GaussianNB()
        self.classifier.fit(self.X_train, self.y_train)
        self.y_predicted = self.classifier.predict(self.X_test)
        return self.classifier
