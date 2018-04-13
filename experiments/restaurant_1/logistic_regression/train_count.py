from os.path import dirname, join
from sklearn.feature_extraction.text import CountVectorizer
from load_data import load_dataset
from model import LogisticRegressionModel

data_train = join(dirname(dirname(dirname(dirname(__file__)))), "data", "vlsp2018", "corpus", "restaurant", "train.xlsx")
data_dev = join(dirname(dirname(dirname(dirname(__file__)))), "data", "vlsp2018", "corpus", "restaurant", "dev.xlsx")

X_train, y_train = load_dataset(data_train)
X_dev, y_dev = load_dataset(data_dev)

for ngram_range in [(1, 2), (1, 3)]:
    for max_features in [500, 700, 800, 900, 1000, 2000, 5000, 7000, 10000, 20000, 30000, 40000]:
    # for max_features in [5000, 7000]:
        name = "Count(ngram {}, max_features {})".format(ngram_range, max_features)
        transformer = CountVectorizer(ngram_range=ngram_range,
                                      max_features=max_features)
        model = LogisticRegressionModel(
            name,
            transformer
        )
        model.load_data(X_train, y_train)
        model.fit_transform()
        model.train()
        model.evaluate(X_dev, y_dev)
    # model.export(folder="exported/svc")
