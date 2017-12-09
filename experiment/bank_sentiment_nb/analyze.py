import json
from os.path import join, dirname
import pandas as pd
import sklearn
from sklearn.preprocessing import MultiLabelBinarizer
from underthesea.util.file_io import write
from bank_sentiment_nb.load_data import load_dataset
from bank_sentiment_nb.model import sentiment


def accuracy_score(TP, FP, TN, FN):
    return round((TP + TN) / (TP + FP + TN + FN), 2)


def precision_score(TP, FP, TN, FN):
    try:
        return round(TP / (TP + FP), 2)
    except:
        return 0


def recall_score(TP, FP, TN, FN):
    try:
        return round(TP / (TP + FN), 2)
    except:
        return 0


def f1_score(TP, FP, TN, FN):
    p = precision_score(TP, FP, TN, FN)
    r = recall_score(TP, FP, TN, FN)
    try:
        f1 = round((2 * p * r) / (p + r), 2)
    except:
        f1 = 0
    return f1


data_file = join(dirname(dirname(dirname(__file__))), "data", "fb_bank_sentiment",
                 "corpus", "test.xlsx")
X_test, y_test = load_dataset(data_file)
y_test = [tuple(item) for item in y_test]
y_pred = sentiment(X_test)

# generate score
labels = set(sum(y_test + y_pred, ()))
score = {}
for label in labels:
    score[label] = {}
    TP, FP, TN, FN = (0, 0, 0, 0)

    for i in range(len(y_test)):
        if label in y_test[i]:
            if label in y_pred[i]:
                TP += 1
            else:
                FN += 1
        else:
            if label in y_pred[i]:
                FP += 1
            else:
                TN += 1
    score[label] = {
        "TP": TP,
        "FP": FP,
        "TN": TN,
        "FN": FN,
        "accuracy": accuracy_score(TP, FP, TN, FN),
        "precision": precision_score(TP, FP, TN, FN),
        "recall": recall_score(TP, FP, TN, FN),
        "f1": f1_score(TP, FP, TN, FN),
    }

df = pd.DataFrame.from_dict(score)
df.T.to_excel("analyze/score.xlsx", columns=["TP", "TN", "FP", "FN", "accuracy", "precision", "recall", "f1"])

# generate result
result = {
    "X_test": X_test,
    "y_test": y_test,
    "y_pred": y_pred,
    "score": score
}

print(score)
content = json.dumps(result, ensure_ascii=False)
write("analyze/result.json", content)

binarizer = MultiLabelBinarizer()
y = [{i for sub in y_test for i in sub}.union({i for sub in y_pred for i in sub})]
binarizer.fit_transform(y)
y_test = binarizer.transform(y_test)
y_pred = binarizer.transform(y_pred)
print("F1 Weighted:", sklearn.metrics.f1_score(y_test, y_pred, average='weighted'))
