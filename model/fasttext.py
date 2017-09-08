import numpy as np
from sklearn.base import ClassifierMixin, BaseEstimator
import fasttext
from underthesea.util.file_io import write
import os


class FastTextClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self):
        self.estimator = None
        pass

    def fit(self, X, y):
        datafile = "temp.train"
        X = [x.replace("\n", " ") for x in X]
        lines = ["__label__{} , {}".format(j, i) for i, j in zip(X, y)]
        content = "\n".join(lines)
        write(datafile, content)
        self.estimator = fasttext.supervised(datafile, 'model_fasttext')
        os.remove(datafile)
        pass

    def predict(self, X):
        return

    def predict_proba(self, X):
        output = np.ones((len(X), 2))
        output_ = self.estimator.predict_proba(X)
        def transform_item(item):
            label, score = item[0]
            label = int(label)
            if label == 0:
                label = 1
                score = 1 - score
            return [label, score]
        output_ = [transform_item(item) for item in output_]
        output1 = np.array(output_)
        return output1
