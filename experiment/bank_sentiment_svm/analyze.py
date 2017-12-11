import json
from os.path import join, dirname
from languageflow.analyze import analyze_multilabel
from underthesea.util.file_io import write
from load_data import load_dataset
from model import sentiment

data_file = join(dirname(dirname(dirname(__file__))), "data", "fb_bank_sentiment",
                 "corpus", "test.xlsx")
X_test, y_test = load_dataset(data_file)
y_test = [tuple(item) for item in y_test]
y_pred = sentiment(X_test)

result = analyze_multilabel(X_test, y_test, y_pred)

content = json.dumps(result, ensure_ascii=False)
write("analyze/result.json", content)
print("F1 Weighted:", result["f1_weighted"])