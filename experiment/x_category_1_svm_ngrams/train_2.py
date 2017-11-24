from os.path import dirname, join

from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from underthesea_flow.flow import Flow
from underthesea_flow.model import Model

from load_data import load_dataset
from transformer import TfidfVectorizer

if __name__ == '__main__':
    data = join(dirname(dirname(dirname(__file__))), "data", "fb_bank_category", "corpus", "data.xlsx")
    X, y = load_dataset(data)

    flow = Flow()
    flow.data(X, y)
    tranformer = TfidfVectorizer(ngram_range=(1, 3))
    flow.transform(MultiLabelBinarizer())
    flow.transform(tranformer)

    flow.add_model(Model(OneVsRestClassifier()))
    pass