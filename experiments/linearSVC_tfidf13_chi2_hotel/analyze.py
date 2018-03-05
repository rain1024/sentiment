from os.path import join, dirname
from languageflow.board import Board
from languageflow.log import MultilabelLogger
from languageflow.log.tfidf import TfidfLogger
from load_data import load_dataset
from model import sentiment

data = join(dirname(dirname(dirname(__file__))), "data", "vlsp2018", "corpus", "test", "hotel.xlsx")
X_test, y_test = load_dataset(data)
y_test = [tuple(item) for item in y_test]
y_pred = sentiment(X_test, y_test)

log_folder = join(dirname(__file__), "analyze")
model_folder = join(dirname(__file__), "model")

board = Board(log_folder=log_folder)

MultilabelLogger.log(X_test, y_test, y_pred, log_folder=log_folder)
TfidfLogger.log(model_folder=model_folder, log_folder=log_folder)

board.serve(port=62010)
