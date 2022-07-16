import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from transformers import TrainingArguments, Trainer, AutoTokenizer, AutoModelForSequenceClassification, pipeline

from data_collection.utils import OTHERS, get_content
from methods.utils import WebDataset


class TransformerClassifier:
    def __init__(
            self, data,
            model_name: str = 'bert-base-cased',
            valtest_size: float = 0.2,
            test_size: float = 0.5
    ):
        self.data = data
        self.X, self.y = self._getXy()
        self.X_train, self.X_testval, self.y_train, self.y_testval = train_test_split(
            self.X, self.y,
            test_size=valtest_size, random_state=1
        )
        self.X_val, self.X_test, self.y_val, self.y_test = train_test_split(
            self.X_testval, self.y_testval,
            test_size=test_size, random_state=1
        )
        self.tokenizer = self._get_tokenizer(model_name)
        self.model = self._get_model(model_name)
        encodings = self._get_encodings(self.tokenizer)
        self.datasets = self._get_datasets(*encodings)

        self.y_predicted = None

    @classmethod
    def _get_tokenizer(cls, model_name):
        return AutoTokenizer.from_pretrained(model_name)

    def _get_model(self, model_name):
        return AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=len(set(self.y))
        )

    def _getXy(self, tokens_key: str = 'tokens'):
        X = [get_content(doc[tokens_key]) for doc in self.data]
        y = np.array([
            doc['category'] if doc['category'] else OTHERS
            for doc in self.data
        ])
        return X, y

    def _get_encodings(self, tokenizer):
        train_encodings = tokenizer(self.X_train, truncation=True, padding=True)
        val_encodings = tokenizer(self.X_val, truncation=True, padding=True)
        test_encodings = tokenizer(self.X_test, truncation=True, padding=True)
        return train_encodings, val_encodings, test_encodings

    def _get_datasets(self, train_encodings, test_encodings, val_encodings):
        train_dataset = WebDataset(train_encodings, self.y_train)
        val_dataset = WebDataset(val_encodings, self.y_val)
        test_dataset = WebDataset(test_encodings, self.y_test)
        return train_dataset, val_dataset, test_dataset

    def _train(self, train_dataset, val_dataset):
        training_args = TrainingArguments(
            output_dir="./results",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=5,
            weight_decay=0.01,
            warmup_steps=500,
            logging_steps=10
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset
        )

        trainer.train()

    def _get_generator(self):
        return pipeline(
            'text-classification', self.model, tokenizer=self.tokenizer)

    def _predict(self, generator):
        return generator([' '.join(x) for x in self.X_test])

    def train(self):
        self._train(*self.datasets[:2])

    def classify(self):
        generator = self._get_generator()
        self.y_predicted = self._predict(generator)

    def f1_score(self):
        return f1_score(self.y_test, self.y_predicted, average='macro')

    def accuracy(self):
        return accuracy_score(self.y_test, self.y_predicted)
