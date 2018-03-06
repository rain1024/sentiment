from os.path import dirname, join
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from load_data import load_dataset
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from score import multilabel_f1_score


data_train = join(dirname(dirname(dirname(__file__))), "data", "vlsp2018", "corpus", "train", "hotel.xlsx")
data_test = join(dirname(dirname(dirname(__file__))), "data", "vlsp2018", "corpus", "test", "hotel.xlsx")

tfidf_ngram12 = TfidfVectorizer(ngram_range=(1, 2))
tfidf_ngram13 = TfidfVectorizer(ngram_range=(1, 3))
tfidf_maxfeature5k = TfidfVectorizer(max_features=5000)

for feature, name in [
    (tfidf_ngram12, 'Tfidf Bigram'),
    (tfidf_ngram13, 'Tfidf Trigram'),
    (tfidf_maxfeature5k, 'Tfidf max features = 5000')
]:
    X, y = load_dataset(data_train)
    X = feature.fit_transform(X)
    transformer = MultiLabelBinarizer()
    y = transformer.fit_transform(y)
    selector = [SelectKBest(chi2, k=i) for i in [1000, 2000, 5000]]
    for select in selector:
        X_train, X_dev, y_train, y_dev = train_test_split(X, y, test_size=0.01)
        X_train = select.fit_transform(X_train, y_train)

        model = OneVsRestClassifier(LinearSVC())
        estimator = model.fit(X_train, y_train)
        y_predict = estimator.predict(select.transform(X_dev))
        score = multilabel_f1_score(y_dev, y_predict)
        print("Feature: ", name)
        print("Score with data train_test_split: ", score)

        X_test, y_test = load_dataset(data_test)
        y_test = [tuple(item) for item in y_test]
        y_pred = transformer.inverse_transform(estimator.predict(select.transform(feature.transform(X_test))))
        score_2 = multilabel_f1_score(y_test, y_pred)
        print("Score with test data: ", score_2)
        print("\n")
