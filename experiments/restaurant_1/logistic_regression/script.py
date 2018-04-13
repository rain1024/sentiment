import os
import ast
import pandas as pd

files = os.listdir("logs")
content = []
for file in files:
    data = open("logs/" + file).read()
    features, score = data.split("\n")
    f1 = "{:.4f}".format(ast.literal_eval(score)["f1"])
    precision = "{:.4f}".format(ast.literal_eval(score)["precision"])
    recall = "{:.4f}".format(ast.literal_eval(score)["recall"])
    max_features = file.split()[5].split(").txt")[0]
    info = [features, f1, precision, recall, max_features]
    content.append(info)
df = pd.DataFrame(content)
df.to_excel("result.xlsx")
