import joblib
from os.path import join, dirname
import sys
from sklearn.feature_selection import chi2, SelectKBest

sys.path.insert(0, dirname(__file__))

y_transform = joblib.load(join(dirname(__file__), "label.transformer.bin"))
x_transform = joblib.load(join(dirname(__file__), "tfidf.transformer.bin"))
estimator = joblib.load(join(dirname(__file__), "model.bin"))
selector = joblib.load(join(dirname(__file__), "selector.transformer.bin"))


def sentiment(X):
    if isinstance(X, list):
        return y_transform.inverse_transform(
            estimator.predict(selector.transform(x_transform.transform(X))))
    else:
        return y_transform.inverse_transform(
            estimator.predict(selector.transform(x_transform.transform(X))))[0]