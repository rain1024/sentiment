from numpy import mean
from sklearn.cross_validation import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, f1_score, make_scorer
from sklearn.preprocessing import MultiLabelBinarizer

from experiment.validation import TrainTestSplitValidation, CrossValidation


class Experiment:
    def __init__(self, X, Y, clf, validation=None):
        self.clf = clf
        self.X = X
        self.Y = Y
        self.validation = validation

    def run(self):
        accuracy_scores = []
        f1_scores = []
        try:
            f1 = []
            accuracy = []

            def score_func(y_true, y_pred, **kwargs):
                accuracy_scores.append(accuracy_score(y_true, y_pred, **kwargs))
                f1_scores.append(f1_score(y_true, y_pred, average='micro', **kwargs))
                return 0

            scorer = make_scorer(score_func)
            if isinstance(self.validation, TrainTestSplitValidation):
                X = self.X.toarray()
                Y = self.Y
                X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=self.validation.test_size)
                self.clf.fit(X_train, Y_train)
                Y_pred = self.clf.predict(X_test)
                score_func(Y_test, Y_pred)
                f1 = mean(f1_scores)
                accuracy = mean(accuracy_scores)
                pass
            elif isinstance(self.validation, CrossValidation):
                cross_val_score(self.clf, self.X.toarray(), self.Y, cv=self.validation.cv, scoring=scorer)
                print("")
                print("F1: {:.4f}".format(mean(f1_scores)))
                print(f1_scores)
                f1.append(mean(f1_scores))
                print("Accuracy: {:.4f}".format(mean(accuracy_scores)))
                print(accuracy_scores)
                accuracy.append(mean(accuracy_scores))
                f1 = mean(f1_scores)
                accuracy = mean(accuracy_scores)
        except Exception as e:
            print(e)
            f1 = 0
            accuracy = 0
        return f1, accuracy

