import joblib
from os.path import join, dirname
import sys

from sklearn.feature_selection import SelectKBest, chi2

sys.path.insert(0, dirname(__file__))

y_transform = joblib.load(join(dirname(__file__), "label.transformer.bin"))
x_transform = joblib.load(join(dirname(__file__), "tfidf.transformer.bin"))
estimator = joblib.load(join(dirname(__file__), "model.bin"))
ch2 = SelectKBest(chi2, k=5000)


def sentiment(X, y):
    if isinstance(X, list):
        return y_transform.inverse_transform(
            estimator.predict(ch2.fit_transform(x_transform.transform(X), y_transform.fit_transform(y))))
    else:
        return y_transform.inverse_transform(
            estimator.predict(ch2.fit_transform(x_transform.transform(X), y_transform.fit_transform(y))))[0]
