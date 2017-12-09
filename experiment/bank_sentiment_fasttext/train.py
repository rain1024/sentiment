from os.path import dirname, join
from languageflow.flow import Flow
from languageflow.model import Model
from languageflow.model.fasttext import FastTextClassifier
from languageflow.transformer.tfidf import TfidfVectorizer
from languageflow.validation.validation import TrainTestSplitValidation
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from bank_sentiment_svm.load_data import load_dataset

if __name__ == '__main__':
    data_file = join(dirname(dirname(dirname(__file__))), "data", "fb_bank_sentiment", "corpus", "train.xlsx")
    X, y = load_dataset(data_file)

    flow = Flow()
    flow.data(X, y)

    # transformer = TfidfVectorizer(ngram_range=(1, 3))
    # flow.transform(MultiLabelBinarizer())
    # flow.transform(transformer)
    # flow.add_model(Model(OneVsRestClassifier(FastTextClassifier()), "FastTextClassifier"))
    flow.add_model(Model(FastTextClassifier(), "FastText"))
    flow.set_validation(TrainTestSplitValidation(test_size=0.1))

    model_name = "FastText"
    model_filename = join("model", "fasttext.model")

    flow.train()
    flow.export(model_name="FastText", export_folder=model_filename)
