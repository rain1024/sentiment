import pandas as pd
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from underthesea.dictionary import Dictionary
from underthesea.feature_engineering.text import Text

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, NuSVC, LinearSVC

from experiment.flow import Flow, Model
from experiment.transformer import TfidfVectorizer, TfidfDictionaryVectorizer

from experiment.validation import TrainTestSplitValidation, CrossValidation


def load_dataset(data_file):
    df = pd.read_excel(data_file)
    X = list(df["text"])

    def convert_text(x):
        try:
            return Text(x)
        except:
            pass
        return ""

    X = [convert_text(x) for x in X]
    Y = df.drop("text", 1)
    # labels = Y.columns
    Y = Y.values
    return X, Y


# data_file = "data/data_3k.xlsx"
# data_file = "data/data_10k.xlsx"
data_file = "data/data.xlsx"
X, Y = load_dataset(data_file)

flow = Flow()

flow.data(X, Y)

# transformer = TfidfVectorizer()
dictionary = Dictionary.Instance()
vocabulary = list(dictionary.words.keys())
# transformer = TfidfVectorizer(ngram_range=(1, 2), max_features=4000)
transformer = TfidfVectorizer(ngram_range=(1, 3), max_features=4000)
# transformer = TfidfVectorizer(default_vocabulary=vocabulary, ngram_range=(1, 3), max_features=4000)
# transformer = TfidfDictionaryVectorizer(default_vocabulary=vocabulary, ngram_range=(1, 3), max_features=4000)
# transformer = TfidfVectorizer(tokenizer=tokenize)
flow.transform(transformer)

# flow.add_model(Model(GaussianNB(), "GaussianNB"))
# flow.add_model(Model(LinearSVC(), "LinearSVC"))

# flow.add_model(Model(OneVsRestClassifier(GaussianNB()), "GaussianNB"))
# flow.add_model(Model(OneVsRestClassifier(NuSVC(nu=0.99)), "NuSVC"))
flow.add_model(Model(OneVsRestClassifier(SVC()), "SVC"))

flow.set_learning_curve(0.7, 1, 0.3)

flow.set_validation(TrainTestSplitValidation(test_size=0.1))
# flow.set_validation(CrossValidation(cv=5))

flow.run()

# flow.train()
