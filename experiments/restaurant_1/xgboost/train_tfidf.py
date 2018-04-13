from os.path import dirname, join
from sklearn.feature_extraction.text import TfidfVectorizer
from load_data import load_dataset
from model import XGboostModel

data_train = join(dirname(dirname(dirname(dirname(__file__)))), "data", "vlsp2018", "corpus", "restaurant", "train.xlsx")
data_dev = join(dirname(dirname(dirname(dirname(__file__)))), "data", "vlsp2018", "corpus", "restaurant", "dev.xlsx")

X_train, y_train = load_dataset(data_train)
X_dev, y_dev = load_dataset(data_dev)

for n_iter in [100, 140, 160, 200, 300]:
    for max_depth in [50, 100, 140, 160, 200, 300, 500]:
    # for max_depth in [200]:
        for ngram_range in [(1, 2), (1, 3)]:
            for max_features in [500, 700, 800, 900, 1000, 2000, 5000, 7000, 10000, 20000, 30000, 40000]:
            # for max_features in [40000]:
                name = "XGBoost(n_iter {0} max_depth {1}) + Tfidf(ngram {2}, max_features {3})".format(n_iter,
                                                                                                       max_depth,
                                                                                                       ngram_range,
                                                                                                       max_features)
                params = {"n_iter": n_iter, "max_depth": max_depth}
                transformer = TfidfVectorizer(ngram_range=ngram_range, max_features=max_features)
                model = XGboostModel(
                    name,
                    params,
                    transformer
                )
                model.load_data(X_train, y_train)
                model.fit_transform()
                model.train()
                model.evaluate(X_dev, y_dev)
    # model.export(folder="exported/svc")
