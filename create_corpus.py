import pandas as pd
from underthesea.util.file_io import read
import json
from functools import reduce


def transform_item(item, labels):
    output = dict()
    output["text"] = item["text"]
    for label in labels:
        output[label] = 1 if label in item["labels"] else 0
    return output


if __name__ == '__main__':
    df = pd.DataFrame()
    data = json.loads(read("data/data.json"))
    labels = list(reduce(lambda x, y: x.union(y), [set(item["labels"]) for item in data]))
    data = [transform_item(item, labels) for item in data]
    df = pd.DataFrame(data, columns=["text"] + labels)
    df.to_excel("data/data.xlsx", index=False, encoding="utf-8")
    print
