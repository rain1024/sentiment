import pandas as pd
from os.path import join
from underthesea.feature_engineering.text import Text
from underthesea_flow.flow import Flow
from underthesea_flow.model import Model
from underthesea_flow.validation.validation import TrainTestSplitValidation

from model.model_fasttext import FastTextClassifier
from models.fb_bank_2_act_fasttext.model_fasttext import FastTextPredictor

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
    y = df.drop("text", 1)
    # labels = Y.columns
    columns = y.columns
    temp = y.apply(lambda _: _ > 0)
    y = list(temp.apply(lambda _: list(columns[_.values]), axis=1))
    return X, y


if __name__ == '__main__':
    # data_file = "corpus/data_3k.xlsx"
    # data_file = "corpus/data_10k.xlsx"
    data_file = "corpus/fb_bank_act_2/data.xlsx"
    X, y = load_dataset(data_file)

    flow = Flow()
    flow.log_folder = "experiments"

    flow.data(X, y)

    flow.add_model(Model(FastTextClassifier(), "FastText"))

    # flow.set_learning_curve(0.7, 1, 0.3)

    flow.set_validation(TrainTestSplitValidation(test_size=0.1))
    # flow.set_validation(CrossValidation(cv=5))

    # flow.validation()

    model_name = "FastText"
    model_filename = join("models", "fb_bank_2_act_fasttext", "fasttext.model")
    flow.train()
    flow.save_model(model_name="FastText", model_filename=model_filename)

    model = FastTextPredictor.Instance()
    X_test, y_test = X, y
    flow.test(X_test, y_test, model)