from os.path import dirname, join
from languageflow.flow import Flow
from languageflow.model import Model
from languageflow.transformer.tfidf import TfidfVectorizer
from languageflow.validation.validation import TrainTestSplitValidation
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC
from load_data import load_dataset


if __name__ == '__main__':
    data_file = join(dirname(dirname(dirname(__file__))), "data", "vlsp2018", "corpus", "train", "resto.xlsx")
    X, y = load_dataset(data_file)

    flow = Flow()
    flow.data(X, y)

    transformer = TfidfVectorizer(ngram_range=(1, 2), min_df=10)
    flow.transform(MultiLabelBinarizer())
    flow.transform(transformer)
    flow.add_model(Model(OneVsRestClassifier(LinearSVC()), "LinearSVC"))
    flow.set_validation(TrainTestSplitValidation(test_size=0.1))

    flow.train()
    flow.export(model_name="LinearSVC", export_folder="model")
