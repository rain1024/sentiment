from os.path import dirname, join
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from load_data import load_dataset
from model import XGboostModel

data_train = join(dirname(dirname(dirname(__file__))), "data", "vlsp2018", "corpus", "train", "hotel.xlsx")
data_test = join(dirname(dirname(dirname(__file__))), "data", "vlsp2018", "corpus", "test", "hotel.xlsx")

X_train, y_train = load_dataset(data_train)
X_test, y_test = load_dataset(data_test)

# n = 100
# X_train, y_train = X_train[:n], y_train[:n]
# X_test, y_test = X_test[:n], y_test[:n]

models = []


# for n_iter in [100, 140, 160, 200, 500]:
for n_iter in [140]:
    # for max_depth in [50, 100, 140, 160, 200, 300]:
    for max_depth in [200, 300, 400, 500]:
        # for max_features in [1000, 2000, 2200, 2400, 2600, 3000]:
        for max_features in [2000, 3000, 4000]:
            name = "XGBoost(n_iter {0} max_depth {1}) + Count(bigram, max_features {2})".format(n_iter, max_depth, max_features)
            params = {"n_iter": n_iter, "max_depth": max_depth}
            model = XGboostModel(
                name,
                params,
                CountVectorizer(ngram_range=(1, 2), max_features=max_features)
            )
            models.append(model)

for model in models:
    from datetime import datetime
    start = datetime.now()
    model.load_data(X_train, y_train)
    model.fit_transform()
    model.train()
    model.evaluate(X_test, y_test)
    # model.export()
    print(datetime.now() - start)
