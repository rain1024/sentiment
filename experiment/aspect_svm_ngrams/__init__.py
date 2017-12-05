from experiment.aspect_svm_ngrams import model


def classify(X, domain=None):
    if X == "":
        return None
    if domain == 'bank':
        return model.aspect(X)
