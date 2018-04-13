from os.path import join, dirname

from exported.svc import sentiment
from load_data import load_dataset
from score import multilabel_f1_score


def _create_log_file_name(score):
    file_name = "logs/" + \
                "{:.4f}".format(score) + \
                "_" + \
                "LinearSVC" + \
                ".txt"
    return file_name


if __name__ == '__main__':
    data_dev = join(dirname(dirname(dirname(__file__))), "data", "vlsp2018", "corpus", "restaurant", "dev.xlsx")
    X_dev, y_dev = load_dataset(data_dev)
    y_pred = sentiment(X_dev)
    score = multilabel_f1_score(y_dev, y_pred)
    print("Score: ", score)
    log_file = _create_log_file_name(score)
    with open(log_file, "w") as f:
        f.write("")
