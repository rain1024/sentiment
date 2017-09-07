from sklearn.feature_extraction import text


class TfidfVectorizer(text.TfidfVectorizer):
    def __init__(self, *args, **kwargs):
        super(TfidfVectorizer, self).__init__(*args, **kwargs)

    def save(self):
        pass
