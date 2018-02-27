from os.path import dirname, join
from languageflow.flow import Flow
from languageflow.model import Model
from languageflow.transformer.count import CountVectorizer
from languageflow.validation.validation import TrainTestSplitValidation
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from load_data import load_dataset


if __name__ == '__main__':
    data_file = join(dirname(dirname(dirname(__file__))), "data", "vlsp2018", "corpus", "train", "hotel.xlsx")
    X, y = load_dataset(data_file)

    flow = Flow()
    flow.data(X, y)

    transformer = CountVectorizer(ngram_range=(1, 2), max_features=5000)
    flow.transform(MultiLabelBinarizer())
    flow.transform(transformer)
    flow.add_model(Model(OneVsRestClassifier(LogisticRegression()), "LogisticRegression"))
    flow.set_validation(TrainTestSplitValidation(test_size=0.0001))

    flow.train()
    flow.export(model_name="LogisticRegression", export_folder="model")