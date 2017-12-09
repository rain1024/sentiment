from os.path import dirname, join
from languageflow.flow import Flow
from languageflow.model import Model
from languageflow.transformer.tfidf import TfidfVectorizer
from languageflow.validation.validation import TrainTestSplitValidation
from sklearn.linear_model import SGDClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.preprocessing import MultiLabelBinarizer
from bank_sentiment_svm.load_data import load_dataset

if __name__ == '__main__':
    data_file = join(dirname(dirname(dirname(__file__))), "data", "fb_bank_sentiment", "corpus", "train.xlsx")
    X, y = load_dataset(data_file)

    flow = Flow()
    flow.data(X, y)

    transformer = TfidfVectorizer(ngram_range=(1, 3))
    flow.transform(MultiLabelBinarizer())
    flow.transform(transformer)

    flow.add_model(Model(OneVsRestClassifier(GaussianNB()), "GaussianNB"))
    # flow.add_model(Model(OneVsRestClassifier(MultinomialNB()), "MultinomialNB"))
    # flow.add_model(Model(OneVsRestClassifier(BernoulliNB()), "BernoulliNB"))

    flow.set_validation(TrainTestSplitValidation(test_size=0.1))

    flow.train()
    flow.export(model_name="GaussianNB", export_folder="model")
    # flow.export(model_name="MultinomialNB", export_folder="model")
    # flow.export(model_name="BernoulliNB", export_folder="model")
