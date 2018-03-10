from os.path import dirname, join
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from load_data import load_dataset
from model import TfidfModel

data_train = join(dirname(dirname(dirname(__file__))), "data", "vlsp2018", "corpus", "train", "hotel.xlsx")
data_test = join(dirname(dirname(dirname(__file__))), "data", "vlsp2018", "corpus", "test", "hotel.xlsx")

X_train, y_train = load_dataset(data_train)
X_test, y_test = load_dataset(data_test)

model = TfidfModel("SVC", CountVectorizer(ngram_range=(1, 2), max_features=900))
model.load_data(X_train, y_train)
model.fit_transform()
model.train()
model.evaluate(X_test, y_test)
model.export(folder="linearsvc_exported")
