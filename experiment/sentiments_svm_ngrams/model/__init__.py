import joblib
from os.path import join, dirname
import sys

sys.path.insert(0, dirname(__file__))

y_transform = joblib.load(join(dirname(__file__), "label.transformer.bin"))
x_transform = joblib.load(join(dirname(__file__), "tfidf.transformer.bin"))
estimator = joblib.load(join(dirname(__file__), "model.bin"))


def sentiment(X):
    if isinstance(X, list):
        y = y_transform.inverse_transform(
            estimator.predict(x_transform.transform(X)))
    else:
        y = y_transform.inverse_transform(
            estimator.predict(x_transform.transform([X])))[0]
    return y
