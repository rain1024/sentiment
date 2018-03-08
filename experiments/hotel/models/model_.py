import joblib
from languageflow.model.xgboost import XGBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from score import multilabel_f1_score
import numpy as np


class Model:
    def __init__(self, name, transformers):
        self.name = name
        self.transformers = transformers

    def get_name(self):
        return self.name

    def load_data(self):
        raise Exception("Need implement")

    def fit_transform(self):
        raise Exception("Need implement")

    def train(self):
        raise Exception("Need implement")

    def evaluate(self):
        raise Exception("Need implement")


class MultilabelXGboostModel(Model):
    def __init__(self, name, transformer):
        self.name = name
        self.y_transformer = LabelEncoder()
        self.transformer = transformer

    def _convert_data_to_multiclass(self):
        X = []
        Y = []
        for i in range(len(self.X)):
            for j in range(len(self.y[i])):
                X.append(self.X[i])
                Y.append(self.y[i][j])
        self.X = X
        self.y = Y

    def load_data(self, X, y):
        self.X = X
        self.y = y
        self._convert_data_to_multiclass()

    def show_info(self):
        print("=======")
        print(self.name)

    def fit_transform(self):
        self.X = self.transformer.fit_transform(self.X)
        self.y = self.y_transformer.fit_transform(self.y)

    def _get_labels(self, Y):
        Y = np.argmax(Y)
        Y = self.y_transformer.inverse_transform(Y)
        Y = [Y]
        return Y

    def _predict(self, X):
        y = self.estimator.predict_proba(X)
        y = [self._get_labels(_) for _ in list(y)]
        return y

    def train(self):
        self.show_info()
        X_train, X_dev, y_train, y_dev = train_test_split(self.X, self.y, test_size=0.01)
        num_classes = max(y_train) + 1
        model = XGBoostClassifier(max_depth=10, num_classes=num_classes, n_jobs=16)
        self.estimator = model.fit(X_train, y_train)
        y_predict = self._predict(X_dev)
        score = multilabel_f1_score(y_dev, y_predict)
        print("Dev Score: ", score)

    def _create_log_file_name(self, score):
        file_name = "logs/" + \
                    "{:.4f}".format(score) + \
                    "_" + \
                    self.name + \
                    ".txt"
        return file_name

    def evaluate(self, X_test, y_test):
        y_test = [tuple(item) for item in y_test]
        y_pred = self.transformer.inverse_transform(
            self.estimator.predict(self.transformer.transform(X_test)))
        score = multilabel_f1_score(y_test, y_pred)
        print("Test Score: ", score)
        log_file = self._create_log_file_name(score)
        with open(log_file, "w") as f:
            f.write("")

    def export(self):
        joblib.dump(self.transformer, "exported/tfidf.transformer.bin")
        joblib.dump(self.y_transformer, "exported/y_transformer.bin")
        joblib.dump(self.estimator, "exported/model.bin", protocol=2)
