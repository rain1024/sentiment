import pandas as pd
from numpy import mean
from sklearn.cross_validation import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import make_scorer, accuracy_score, f1_score
from sklearn.naive_bayes import GaussianNB
from underthesea.feature_engineering.text import Text
from underthesea.word_sent.tokenize import tokenize

df = pd.read_excel("data/data.xlsx")
X = list(df["text"])


def convert_text(x):
    try:
        return Text(x)
    except:
        pass
    return ""


X = [convert_text(x) for x in X]
# transformer = TfidfVectorizer(tokenizer=tokenize)
transformer = TfidfVectorizer()
# transformer = TfidfVectorizer(min_df=0.4, tokenizer=tokenize)
X = transformer.fit_transform(X)

Y = list(df["PRICE"])
# Y = list(df["SPEED_BANDWIDTH"])
clf = GaussianNB()

accuracy_scores = []
f1_scores = []


def get_scores():
    return accuracy_scores, f1_scores


def score_func(y_true, y_pred, **kwargs):
    accuracy_scores.append(accuracy_score(y_true, y_pred, **kwargs))
    f1_scores.append(f1_score(y_true, y_pred, **kwargs))
    return 0


scorer = make_scorer(score_func)
cross_val_score(clf, X.toarray(), Y, cv=3, scoring=scorer)
print(sum(Y))
print(f1_scores)
print("{:.4f}".format(mean(f1_scores)))
print(accuracy_scores)
print("{:.4f}".format(mean(accuracy_scores)))
