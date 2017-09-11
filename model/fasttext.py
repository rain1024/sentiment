import random
from os.path import join

import numpy as np
from sklearn.base import ClassifierMixin, BaseEstimator
import fasttext as ft
from underthesea.util.file_io import write
import os


class FastTextClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self):
        self.estimator = None

    def fit(self, X, y):
        """Fit FastText according to X, y

        Parameters:
        ----------
        X : list of text
            each item is a text
        y: list
           each item is either a label (in multi class problem) or list of
           labels (in multi label problem)
        """
        datafile = "temp.train"
        X = [x.replace("\n", " ") for x in X]
        lines = ["__label__{} , {}".format(j, i) for i, j in zip(X, y)]
        content = "\n".join(lines)
        write(datafile, content)
        self.state = "{}".format(random.randint(1, 1000000))
        bin_file = join("fasttext_bin", self.state)
        self.estimator = ft.supervised(datafile, bin_file)
        os.remove(datafile)

    def __getstate__(self):
        try:
            print("Get State:", self.state)
            return self.state
        except:
            return 0

    def __setstate__(self, state):
        print("Set state:", state)
        bin_file = join("fasttext_bin", "{}.bin".format(state))
        try:
            self.estimator = ft.load_model(bin_file)
        except:
            self.estimator = None

    def predict(self, X):
        return

    def predict_proba(self, X):
        output = np.ones((len(X), 2))
        output_ = self.estimator.predict_proba(X)

        def transform_item(item):
            label, score = item[0]
            label = label.replace("__label__", "")
            label = int(label)
            if label == 0:
                label = 1
                score = 1 - score
            return [label, score]

        output_ = [transform_item(item) for item in output_]
        output1 = np.array(output_)
        return output1
