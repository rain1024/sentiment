from os.path import join, dirname
import sys
import joblib
sys.path.insert(0, dirname(__file__))


def sentiment(text):
    model = joblib.load(join(dirname(__file__), "model.bin"))
    y = model.predict(text)
    y = list(y)
    return y