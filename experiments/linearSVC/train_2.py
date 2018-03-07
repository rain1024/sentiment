from os.path import dirname, join
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from load_data import load_dataset
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from score import multilabel_f1_score
from model import Model, TfidfModel

data_train = join(dirname(dirname(dirname(__file__))), "data", "vlsp2018", "corpus", "train", "hotel.xlsx")
data_test = join(dirname(dirname(dirname(__file__))), "data", "vlsp2018", "corpus", "test", "hotel.xlsx")

X_train, y_train = load_dataset(data_train)
X_test, y_test = load_dataset(data_test)

# n = 10
# X_train, y_train = X_train[:n], y_train[:n]
# X_test, y_test = X_test[:n], y_test[:n]

# parameters = {
#     'max_features': (None, 5000, 10000, 15000),
#     'ngram_range': ((1, 2), (1, 3)),  # bigrams or trigrams
# }
# pipeline = Pipeline([
#     ('vect', CountVectorizer()),
#     ('tfidf', TfidfVectorizer()),
# ])
# models = LinearSVC(pipeline, parameters)

models = [
    # TfidfModel("Tfidf Bigram", TfidfVectorizer(ngram_range=(1, 2))),
    # TfidfModel("Tfidf Trigram", TfidfVectorizer(ngram_range=(1, 3))),
    # TfidfModel("Tfidf Max Feature", TfidfVectorizer(max_features=5000)),
    # TfidfModel("Count Bigram", CountVectorizer(ngram_range=(1, 2))),
    # TfidfModel("Count Trigram", CountVectorizer(ngram_range=(1, 3)))
]

# for n in [2000, 5000, 10000, 15000, 20000]:
#     model = TfidfModel(
#         "Count Max Feature {}".format(n),
#         CountVectorizer(max_features=n)
#     )
#     models.append(model)

for n in [500, 700, 800, 900, 1000]:
    for ngram in [('Bigram', (1, 2)), ("Trigram", (1, 3))]:
        model = TfidfModel(
            "Count {0} + Max Feature {1}".format(ngram[0], n),
            CountVectorizer(ngram_range=ngram[1], max_features=n)
        )
        models.append(model)

for model in models:
    model.load_data(X_train, y_train)
    model.fit_transform()
    model.train()
    model.evaluate(X_test, y_test)
