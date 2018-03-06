import joblib
from score import multilabel_f1_score

# y_true = joblib.load('y_true')
# y_pred = joblib.load('y_pred')
#
# score = multilabel_f1_score(y_true, y_pred)
# print(score)
#
# y_true = [[0, 0, 1]]
# y_pred = [[0, 0, 1]]
# print(multilabel_f1_score(y_true, y_pred))

# y_true = [[0, 1, 1]]
# y_pred = [[0, 0, 1]]
# print(multilabel_f1_score(y_true, y_pred))

# y_true = [[0, 0, 1]]
# y_pred = [[0, 1, 1]]
# print(multilabel_f1_score(y_true, y_pred))

y_true = [[0, 0, 0]]
y_pred = [[0, 0, 0]]
print(multilabel_f1_score(y_true, y_pred))
