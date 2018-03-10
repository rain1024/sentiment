from os.path import dirname, join
import re
from languageflow.util.file_io import read
import pandas as pd
import re

def transform(s):
    sentence = {}
    sentence["text"] = s.split("\n")[1]
    sentiments = s.split("\n")[2]
    sentiments_ = re.split("}, +{", sentiments)
    sentiments__ = [re.sub(r"[{}]", "", item) for item in sentiments_]
    labels = [item.upper().replace(", ", "#") for item in sentiments__]
    sentence["labels"] = labels
    return sentence


def convert_to_corpus(sentences, corpus_file):
    data = []
    labels = list(set(sum([s["labels"] for s in sentences], [])))
    for s in sentences:
        item = {}
        item["text"] = s["text"]
        for label in labels:
            if label in s["labels"]:
                item[label] = 1
            else:
                item[label] = 0
        data.append(item)
    df = pd.DataFrame(data)
    columns = ["text"] + labels
    df.to_excel(corpus_file, index=False, columns=columns)


if __name__ == '__main__':
    data = read(join(dirname(__file__), "raw", "hotel", "1-VLSP2018-SA-hotel-train (7-3-2018).txt")).split("\n\n")
    corpus_file = join(dirname(__file__), "corpus", "hotel", "train.xlsx")
    sentences = [transform(item) for item in data]
    convert_to_corpus(sentences, corpus_file)

    sentences = read(join(dirname(__file__), "raw", "hotel", "2-VLSP2018-SA-hotel-dev (7-3-2018).txt")).split("\n\n")
    corpus_file = join(dirname(__file__), "corpus", "hotel", "dev.xlsx")
    sentences = [transform(s) for s in sentences]
    convert_to_corpus(sentences, corpus_file)

