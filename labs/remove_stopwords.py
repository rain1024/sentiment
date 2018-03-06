from os.path import join, dirname
import pandas as pd
from languageflow.util.file_io import read

data_file = join(dirname(dirname(__file__)), "data", "vlsp2018", "corpus", "train", "hotel.xlsx")
stop_words = read(join(dirname(__file__), "vi_stopwords.txt")).split("\n")
df = pd.read_excel(data_file)
X = list(df["text"])
for text in X:
    comment = " ".join([w for w in text.lower().split() if not w in set(stop_words)])
print(0)
