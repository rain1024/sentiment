import pandas as pd
from os import listdir

from os.path import join, dirname

from underthesea.feature_engineering.text import Text
from underthesea.util.file_io import read
import json
from functools import reduce


def read_utf16(filename):
    with open(filename, 'rb') as f:
        content = f.read()
        content = content.decode("utf-16")
        return Text(content)


def transform_text(text, label, labels):
    output = {}
    output["text"] = text
    output["label"] = 1
    for l in labels:
        output[l] = 1 if l == label else 0
    return output


if __name__ == '__main__':
    # train dataset
    FOLDER = dirname(__file__)
    data_folder = join(FOLDER, "data", "Train_Full")
    labels = listdir(data_folder)
    dataset = []
    for label in labels:
        folder = join(data_folder, label)
        files = listdir(folder)
        texts = [read_utf16(join(folder, file)) for file in files]
        items = [transform_text(text, label, labels) for text in texts]
        dataset += items
    columns = ["text"] + labels
    df = pd.DataFrame(dataset, columns=columns)
    df.to_excel("data/data.xlsx", index=False, encoding="utf-8")
    print(0)
