from os.path import dirname, join
from sklearn import feature_extraction

from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from underthesea_flow.flow import Flow
from underthesea_flow.model import Model
from underthesea_flow.transformer.tfidf import TfidfVectorizer
from underthesea_flow.validation.validation import TrainTestSplitValidation
from sklearn.preprocessing import MultiLabelBinarizer
from load_data import load_dataset


class CustomVectorizer(feature_extraction.text.TfidfVectorizer):
    def __init__(self, *args, **kwargs):
        super(CustomVectorizer, self).__init__(*args, **kwargs)

    def hasNumbers(self, s):
        return any(c.isdigit() for c in s)

    def text2vec(self, text):
        vectorizer = feature_extraction.text.TfidfVectorizer(ngram_range=(1, 3))
        vectorizer.fit_transform(text)
        vocab = [item for item in set(vectorizer.vocabulary_) if
                 not self.hasNumbers(item)]
        vectorizer = feature_extraction.text.TfidfVectorizer(vocabulary=vocab,
                                                             ngram_range=(1, 3))
        vectorizer.fit_transform(text)
        self = vectorizer
        return vectorizer.fit_transform(text)

if __name__ == '__main__':
    data_file = join(dirname(dirname(dirname(__file__))), "data", "fb_bank_act_3", "corpus", "train.xlsx")
    X, y = load_dataset(data_file)

    flow = Flow()
    flow.log_folder = "log"

    flow.data(X, y)

    transformer = CustomVectorizer()
    flow.transform(transformer)
    flow.transform(MultiLabelBinarizer())


    flow.add_model(Model(OneVsRestClassifier(LogisticRegression()), "LogisticRegression"))

    # flow.set_learning_curve(0.7, 1, 0.3)
    flow.set_validation(TrainTestSplitValidation(test_size=0.1))

    flow.train()
    flow.export_folder = "model"
    flow.export(model_name="LogisticRegression")