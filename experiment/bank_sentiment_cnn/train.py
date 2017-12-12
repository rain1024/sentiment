from os.path import dirname, join
from languageflow.flow import Flow
from languageflow.model import Model
from languageflow.model.cnn import KimCNNClassifier
from languageflow.validation.validation import TrainTestSplitValidation
from load_data import load_dataset
from sklearn.preprocessing import LabelEncoder

if __name__ == '__main__':
    data_file = join(dirname(dirname(dirname(__file__))), "data", "fb_bank_sentiment", "corpus", "train.xlsx")
    X, y = load_dataset(data_file)

    flow = Flow()
    flow.log_folder = "log"
    flow.data(X, y)

    flow.add_model(Model(KimCNNClassifier(batch_size=5, epoch=150, embedding_dim=300), "KimCNNClassifier"))
    flow.set_validation(TrainTestSplitValidation(test_size=0.1))

    # flow.train()

    export_folder = join(dirname(__file__), "model")
    flow.export(model_name="KimCNNClassifier", export_folder=export_folder)
