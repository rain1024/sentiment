import joblib
from os.path import join

from languageflow.model.xgboost import XGBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
import json

from score import multilabel_score


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


class SVCModel(Model):
    def __init__(self, name, transformer):
        self.name = name
        self.y_transformer = MultiLabelBinarizer()
        self.transformer = transformer

    def load_data(self, X, y):
        self.X = X
        self.y = y

    def show_info(self):
        print("=======")
        print(self.name)

    def fit_transform(self):
        self.X = self.transformer.fit_transform(self.X)
        self.y = self.y_transformer.fit_transform(self.y)

    def train(self):
        self.show_info()
        model = OneVsRestClassifier(LinearSVC())
        self.estimator = model.fit(self.X, self.y)

    def _create_log_file_name(self, score):
        file_name = "logs/" + \
                    "{:.4f}".format(score["f1"]) + \
                    "_" + \
                    "LinearSVC" + \
                    "_" + \
                    self.name + \
                    ".txt"
        return file_name

    def evaluate(self, X_test, y_test):
        y_test = [tuple(item) for item in y_test]
        y_pred = self.y_transformer.inverse_transform(
            self.estimator.predict(self.transformer.transform(X_test)))
        score = multilabel_score(y_test, y_pred)
        print("Score: ", score)
        log_file = self._create_log_file_name(score)
        with open(log_file, "w") as f:
            content = self.name + "\n" + json.dumps(score)
            f.write(content)

    def export(self, folder="exported"):
        joblib.dump(self.transformer,
                    join(folder, "count.transformer.bin"))
        joblib.dump(self.y_transformer,
                    join(folder, "y_transformer.bin"))
        joblib.dump(self.estimator,
                    join(folder, "model.bin"),
                    protocol=2)


class LogisticRegressionModel(Model):
    def __init__(self, name, transformer):
        self.name = name
        self.y_transformer = MultiLabelBinarizer()
        self.transformer = transformer

    def load_data(self, X, y):
        self.X = X
        self.y = y

    def show_info(self):
        print("=======")
        print(self.name)

    def fit_transform(self):
        self.X = self.transformer.fit_transform(self.X)
        self.y = self.y_transformer.fit_transform(self.y)

    def train(self):
        self.show_info()
        model = OneVsRestClassifier(LogisticRegression())
        self.estimator = model.fit(self.X, self.y)

    def _create_log_file_name(self, score):
        file_name = "logs/" + \
                    "{:.4f}".format(score["f1"]) + \
                    "_" + \
                    "Logistic Regression" + \
                    "_" + \
                    self.name + \
                    ".txt"
        return file_name

    def evaluate(self, X_test, y_test):
        y_test = [tuple(item) for item in y_test]
        y_pred = self.y_transformer.inverse_transform(
            self.estimator.predict(self.transformer.transform(X_test)))
        score = multilabel_score(y_test, y_pred)
        print("Score: ", score)
        log_file = self._create_log_file_name(score)
        with open(log_file, "w") as f:
            content = self.name + "\n" + json.dumps(score)
            f.write(content)


class GaussianModel(Model):
    def __init__(self, name, transformer):
        self.name = name
        self.y_transformer = MultiLabelBinarizer()
        self.transformer = transformer

    def load_data(self, X, y):
        self.X = X
        self.y = y

    def show_info(self):
        print("=======")
        print(self.name)

    def fit_transform(self):
        self.X = self.transformer.fit_transform(self.X)
        self.X = self.X.toarray()
        self.y = self.y_transformer.fit_transform(self.y)

    def train(self):
        self.show_info()
        model = OneVsRestClassifier(GaussianNB())
        self.estimator = model.fit(self.X, self.y)

    def _create_log_file_name(self, score):
        file_name = "logs/" + \
                    "{:.4f}".format(score["f1"]) + \
                    "_" + \
                    "Gaussian" + \
                    "_" + \
                    self.name + \
                    ".txt"
        return file_name

    def evaluate(self, X_test, y_test):
        y_test = [tuple(item) for item in y_test]
        y_pred = self.y_transformer.inverse_transform(
            self.estimator.predict(self.transformer.transform(X_test).toarray()))
        score = multilabel_score(y_test, y_pred)
        print("Score: ", score)
        log_file = self._create_log_file_name(score)
        with open(log_file, "w") as f:
            content = self.name + "\n" + json.dumps(score)
            f.write(content)


class XGboostModel(Model):
    def __init__(self, name, params, transformer):
        self.name = name
        self.y_transformer = MultiLabelBinarizer()
        self.transformer = transformer
        self.params = params
        self.model = OneVsRestClassifier(XGBoostClassifier(**params))

    def load_data(self, X, y):
        self.X = X
        self.y = y

    def show_info(self):
        print("=======")
        print(self.name)

    def fit_transform(self):
        self.X = self.transformer.fit_transform(self.X)
        self.y = self.y_transformer.fit_transform(self.y)

    def train(self):
        self.show_info()
        self.estimator = self.model.fit(self.X, self.y)

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
        score = multilabel_score(y_test, y_pred)
        print("Test Score: ", score)
        log_file = self._create_log_file_name(score)
        with open(log_file, "w") as f:
            f.write("")
