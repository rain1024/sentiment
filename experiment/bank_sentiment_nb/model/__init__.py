import joblib
from os.path import join, dirname
import sys
sys.path.insert(0, dirname(__file__))

y_transform = joblib.load(join(dirname(__file__), "label.transformer.bin"))
x_transform = joblib.load(join(dirname(__file__), "tfidf.transformer.bin"))
estimator = joblib.load(join(dirname(__file__), "model.bin"))


# def sentiment(X):
#     if not isinstance(X, list):
#         x = [X]
#     x = x_transform.transform(x)
#     x = x.toarray()
#     y = estimator.predict(x)
#     y = y_transform.inverse_transform(y)
#     if not isinstance(X, list):
#         y = y[0]
#     return y

def sentiment(X):
    if not isinstance(X, list):
        x = [X]
    else:
        x = X
    x = x_transform.transform(x)
    x = x.toarray()
    y = estimator.predict(x)
    y = y_transform.inverse_transform(y)
    if not isinstance(X, list):
        y = y[0]
    return y
