from os.path import dirname, join
from languageflow.flow import Flow
from languageflow.model import Model
from languageflow.model.fasttext import FastTextClassifier
from languageflow.validation.validation import TrainTestSplitValidation
from load_data import load_dataset
from sklearn.preprocessing import LabelEncoder

if __name__ == '__main__':
    data_file = join(dirname(dirname(dirname(__file__))), "data", "fb_bank_sentiment", "corpus", "train.xlsx")
    X, y = load_dataset(data_file)

    flow = Flow()
    flow.log_folder = "log"
    flow.data(X, y)

    # transformer = LabelEncoder()
    # flow.transform(transformer)
    flow.add_model(Model(FastTextClassifier(), "FastText"))
    flow.set_validation(TrainTestSplitValidation(test_size=0.1))

    flow.train()

    export_folder = join(dirname(__file__), "model")
    flow.export(model_name="FastText", export_folder=export_folder)
