from os.path import join, dirname
import sys
from languageflow import api
sys.path.insert(0, dirname(__file__))


def sentiment(text):
    model = api.load("FastText", join(dirname(__file__), "model.bin.bin"))
    y = model.predict(text)
    return y