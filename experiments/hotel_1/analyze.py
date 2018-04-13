from os.path import join, dirname
from exported.svc import sentiment
from load_data import load_dataset


def get_aspect(y):
    labels = [i for item in y for i in item]
    labels = ["#".join(i.split("#")[0:2]) for i in labels]
    aspect = set(labels)
    return aspect


data = join(dirname(dirname(dirname(__file__))), "data", "vlsp2018", "corpus", "hotel", "dev.xlsx")
X_dev, y_dev = load_dataset(data)
import joblib

# y_pred = sentiment(X_dev)
# joblib.dump(y_pred, "y_pred.bin")
y_pred = joblib.load("y_pred.bin")

aspects = [
    "ROOM_AMENITIES#CLEANLINESS",
    "SERVICE#GENERAL",
    "ROOMS#CLEANLINESS",
    "ROOMS#COMFORT",
    "LOCATION#GENERAL",
    "ROOMS#GENERAL",
    "ROOMS#DESIGN&FEATURES",
    "HOTEL#CLEANLINESS",
    "ROOM_AMENITIES#COMFORT",
    "ROOM_AMENITIES#DESIGN&FEATURES",
    "ROOM_AMENITIES#GENERAL",
    "FOOD&DRINKS#STYLE&OPTIONS",
    "ROOMS#QUALITY",
    "FACILITIES#DESIGN&FEATURES",
    "HOTEL#DESIGN&FEATURES",
    "FACILITIES#QUALITY",
    "HOTEL#QUALITY",
    "HOTEL#PRICES",
    "HOTEL#GENERAL",
    "ROOMS#PRICES",
    "HOTEL#COMFORT",
    "FACILITIES#GENERAL",
    "HOTEL#MISCELLANEOUS",
    "ROOM_AMENITIES#QUALITY",
    "FACILITIES#MISCELLANEOUS",
    "FACILITIES#COMFORT",
    "FOOD&DRINKS#QUALITY",
    "FOOD&DRINKS#MISCELLANEOUS",
    "FACILITIES#PRICES",
    "FOOD&DRINKS#PRICES",
    "FACILITIES#CLEANLINESS",
    "ROOMS#MISCELLANEOUS",
    "ROOM_AMENITIES#MISCELLANEOUS",
    "ROOM#MISCELLANEOUS",
    "ROOM_AMENITIES#PRICES",
]

scores = {}
for aspect in aspects:
    scores[aspect] = {
        "gold": 0,
        "answer": 0,
        "correct_aspect": 0,
        "precision_aspect": 0,
        "recall_aspect": 0,
        "f1_aspect": 0,
        "correct_sentiment": 0,
        "precision_sentiment": 0,
        "f1_sentiment": 0,
    }


def extract_aspect(label):
    return label[:label.rfind("#")]


import pandas as pd

for i in range(len(y_dev)):
    dev = y_dev[i]
    pred = y_pred[i]
    for aspect in aspects:
        gold_count = len([label for label in dev if aspect in label])
        answer_count = len([label for label in pred if aspect in label])
        scores[aspect]["gold"] += gold_count
        scores[aspect]["answer"] += answer_count
    dev_aspects = [extract_aspect(label) for label in dev]
    pred_aspects = [extract_aspect(label) for label in pred]
    correct_aspects = set(dev_aspects).intersection(set(pred_aspects))
    for aspect in correct_aspects:
        scores[aspect]["correct_aspect"] += 1

    correct_sentiments = set(dev).intersection(set(pred))
    for label in correct_sentiments:
        aspect = extract_aspect(label)
        scores[aspect]["correct_sentiment"] += 1


def calculate_scores(correct, answer, gold):
    try:
        p = correct / answer
    except:
        p = 0
    try:
        r = correct / gold
    except:
        r = 0
    try:
        f1 = (2 * p * r) / (p + r)
    except:
        f1 = 0
    return p, r, f1

total_answer = 0
total_gold = 0
total_correct_aspect = 0
total_correct_sentiment= 0

for aspect in aspects:
    answer = scores[aspect]["answer"]
    gold = scores[aspect]["gold"]
    p, r, f1 = calculate_scores(scores[aspect]["correct_aspect"], answer, gold)
    scores[aspect]["precision_aspect"] = p
    scores[aspect]["recall_aspect"] = r
    scores[aspect]["f1_aspect"] = f1

    p, r, f1 = calculate_scores(scores[aspect]["correct_sentiment"], answer, gold)
    scores[aspect]["precision_sentiment"] = p
    scores[aspect]["recall_sentiment"] = r
    scores[aspect]["f1_sentiment"] = f1
    total_answer += answer
    total_gold += gold
    total_correct_aspect += scores[aspect]["correct_aspect"]
    total_correct_sentiment += scores[aspect]["correct_sentiment"]

print(calculate_scores(total_correct_aspect, total_answer, total_gold))
print(calculate_scores(total_correct_sentiment, total_answer, total_gold))
df = pd.DataFrame(scores, columns=aspects)
df.to_excel("scores.xlsx")
