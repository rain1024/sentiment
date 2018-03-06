from os.path import dirname, join
import re
from languageflow.util.file_io import read
import pandas as pd


def transform_comment(item):
    comment = {}
    comment["text"] = item.split("\n")[1]
    sentiments = item.split("\n")[2]
    sentiments = [re.sub(r"[{}]", "", item) for item in sentiments.split("}, {")]
    comment["labels"] = [item.upper().replace(", ", "#") for item in sentiments]
    return comment


def convert_to_corpus(comments, corpus_file):
    data = []
    labels = list(set(sum([comment["labels"] for comment in comments], [])))
    for comment in comments:
        item = {}
        item["text"] = comment["text"]
        for label in labels:
            if label in comment["labels"]:
                item[label] = 1
            else:
                item[label] = 0
        data.append(item)
    df = pd.DataFrame(data)
    columns = ["text"] + labels
    df.to_excel(corpus_file, index=False, columns=columns)


if __name__ == '__main__':
    data = read(join(dirname(__file__), "raw", "train", "1-VLSP2018-SA-hotel-train (4-3-2018).txt")).split("\n\n")
    corpus_file = join(dirname(__file__), "corpus", "train", "hotel.xlsx")
    comments = [transform_comment(item) for item in data]
    convert_to_corpus(comments, corpus_file)

    data = read(join(dirname(__file__), "raw", "dev", "2-VLSP2018-SA-hotel-dev (4-3-2018).txt")).split("\n\n")
    corpus_file = join(dirname(__file__), "corpus", "test", "hotel.xlsx")
    comments = [transform_comment(item) for item in data]
    convert_to_corpus(comments, corpus_file)
