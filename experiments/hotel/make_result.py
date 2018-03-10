from os.path import join, dirname
from languageflow.board import Board
from languageflow.log import MultilabelLogger
import pandas as pd
import re
from linearsvc_exported import sentiment

data = join(dirname(dirname(dirname(__file__))), "data", "vlsp2018", "corpus", "test", "hotel-test.xlsx")
X_test = list(pd.read_excel(data)["text"])

y = sentiment(X_test)


def generate_labels(y):
    labels = []
    for item in y:
        matched = re.match("^(?P<attribute>.*)#(?P<sentiment>POSITIVE|NEGATIVE|NEUTRAL)$", item)
        attribute = matched.group("attribute")
        sentiment = matched.group("sentiment")
        label = "{}, {}".format(attribute, sentiment.lower())
        label = "{" + label + "}"
        labels.append(label)
    labels = ", ".join(labels)
    return labels


content = ""
for i in range(len(X_test)):
    content += "#{}\n".format(i + 1)
    content += "{}\n".format(X_test[i])
    content += "{}\n\n".format(generate_labels(y[i]))

open("results/result.txt", "w").write(content)
