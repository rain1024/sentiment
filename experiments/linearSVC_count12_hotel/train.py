from os.path import dirname, join
import joblib
from languageflow.transformer.count import CountVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from load_data import load_dataset
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from score import multilabel_f1_score

if __name__ == '__main__':
    data_file = join(dirname(dirname(dirname(__file__))), "data", "vlsp2018", "corpus", "train", "hotel.xlsx")
    X, y = load_dataset(data_file)

    transformer_1 = CountVectorizer(ngram_range=(1, 2))
    X = transformer_1.fit_transform(X)
    transformer_2 = MultiLabelBinarizer()
    y = transformer_2.fit_transform(y)

    X_train, X_dev, y_train, y_dev = train_test_split(X, y, test_size=0.01)
    model = OneVsRestClassifier(LinearSVC())
    model.fit(X_train, y_train)
    y_predict = model.predict(X_dev)
    score = multilabel_f1_score(y_dev, y_predict)

    joblib.dump(transformer_2, join("model", "label.transformer.bin"))
    joblib.dump(transformer_1, join("model", "count.transformer.bin"))
    estimator = OneVsRestClassifier(LinearSVC()).fit(X_train, y_train)
    joblib.dump(estimator, join("model", "model.bin"), protocol=2)
    print(score)
