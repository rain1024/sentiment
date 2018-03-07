from os.path import dirname, join
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from load_data import load_dataset
from model import XGboostModel

data_train = join(dirname(dirname(dirname(__file__))), "data", "vlsp2018", "corpus", "train", "restaurant.xlsx")
data_test = join(dirname(dirname(dirname(__file__))), "data", "vlsp2018", "corpus", "test", "restaurant.xlsx")

X_train, y_train = load_dataset(data_train)
X_test, y_test = load_dataset(data_test)

models = [
    XGboostModel("Tfidf Bigram", TfidfVectorizer(ngram_range=(1, 2))),
    # XGboostModel("Tfidf Trigram", TfidfVectorizer(ngram_range=(1, 3))),
    # XGboostModel("Count Bigram", CountVectorizer(ngram_range=(1, 2))),
    # XGboostModel("Count Trigram", CountVectorizer(ngram_range=(1, 3)))
]

# for n in [2000, 5000, 10000, 15000, 20000]:
#     model = XGboostModel(
#         "Count Max Feature {}".format(n),
#         CountVectorizer(max_features=n)
#     )
#     models.append(model)

# for n in [2000, 5000, 10000, 15000, 20000]:
#     model = XGboostModel(
#         "Count Max Feature {}".format(n),
#         TfidfVectorizer(max_features=n)
#     )
#     models.append(model)

# for n in [500, 700, 800, 900, 1000]:
#     for ngram in [('Bigram', (1, 2)), ("Trigram", (1, 3))]:
#         model = XGboostModel(
#             "Count {0} + Max Feature {1}".format(ngram[0], n),
#             CountVectorizer(ngram_range=ngram[1], max_features=n)
#         )
#         models.append(model)

# for n in [500, 700, 800, 900, 1000]:
#     for ngram in [('Bigram', (1, 2)), ("Trigram", (1, 3))]:
#         model = XGboostModel(
#             "Count {0} + Max Feature {1}".format(ngram[0], n),
#             TfidfVectorizer(ngram_range=ngram[1], max_features=n)
#         )
#         models.append(model)

for model in models:
    model.load_data(X_train, y_train)
    model.fit_transform()
    for n in [10, 20, 30, 50]:
        model.train(n)
        model.evaluate(X_test, y_test)
