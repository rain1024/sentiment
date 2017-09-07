import numpy
import pandas as pd
from numpy import mean
from sklearn.cross_validation import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import make_scorer, accuracy_score, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, NuSVC, LinearSVC
from underthesea.feature_engineering.text import Text
from underthesea.word_sent.tokenize import tokenize

df = pd.read_excel("data/data.xlsx")
X = list(df["text"])


def convert_text(x):
    try:
        return Text(x)
    except:
        pass
    return ""


X = [convert_text(x) for x in X]
transformer = TfidfVectorizer()
# transformer = TfidfVectorizer(min_df=0.4, tokenizer=tokenize)
# transformer = TfidfVectorizer(tokenizer=tokenize)
X = transformer.fit_transform(X)

Y = list(df["SPEED_BANDWIDTH"])
# Y = list(df["PRICE"])

import matplotlib.pyplot as plt


class ExperimentLab:
    def __init__(self):
        self.experiments = []

    def add(self, experiment):
        self.experiments.append(experiment)

    def run(self):
        ir = numpy.arange(0.4, 1, 0.05)
        N = [int(i * len(Y)) for i in ir]
        clfs = [SVC(), NuSVC(), GaussianNB()]
        linestyles = ['-', '--', 'dotted']
        models = ["SVC", "NuSVC", "GaussianNB"]
        legends = []
        for i, clf in enumerate(clfs):
            f1_scores = []
            accuracy_scores = []
            for n in N:
                e = DataExperiment(n, clf)
                f1, accuracy = e.run()
                f1_scores.append(f1)
                accuracy_scores.append(accuracy)
            plt.gca().set_color_cycle(['red', 'green'])
            plt.plot(N, f1_scores, ls=linestyles[i])
            plt.plot(N, accuracy_scores, ls=linestyles[i])
            legends.append("{} f1".format(models[i]))
            legends.append("{} accuracy".format(models[i]))
        plt.legend(legends)
        plt.xlabel("Train size")
        plt.ylabel("Score")
        plt.savefig("learning_curve.png")
        plt.show()

    def visualize(self):
        pass


class DataExperiment:
    def __init__(self, n, clf):
        self.n = n
        self.clf = clf

    def run(self):
        X_ = X[:self.n]
        Y_ = Y[:self.n]
        accuracy_scores = []
        f1_scores = []
        try:
            f1 = []
            accuracy = []

            def score_func(y_true, y_pred, **kwargs):
                accuracy_scores.append(accuracy_score(y_true, y_pred, **kwargs))
                f1_scores.append(f1_score(y_true, y_pred, **kwargs))
                return 0

            scorer = make_scorer(score_func)
            cross_val_score(self.clf, X_.toarray(), Y_, cv=3, scoring=scorer)
            print(f1_scores)
            print("F1: {:.4f}".format(mean(f1_scores)))
            f1.append(mean(f1_scores))
            print(accuracy_scores)
            print("Accuracy: {:.4f}".format(mean(accuracy_scores)))
            accuracy.append(mean(accuracy_scores))
            f1 = mean(f1_scores)
            accuracy = mean(accuracy_scores)
        except:
            f1 = 0
            accuracy = 0
        return f1, accuracy


lab = ExperimentLab()
lab.run()
