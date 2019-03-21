import pandas as pd

# https://machinelearningmastery.com/prepare-text-data-machine-learning-scikit-learn/
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html
# https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html#sklearn.feature_extraction.text.TfidfTransformer
# https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
# https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import HashingVectorizer

# CountVectorizer: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
def countvectorize(corpus):
    v = CountVectorizer()
    counts = v.fit_transform((corpus))
    t = TfidfTransformer().fit(counts)
    return counts
    # return t.transform(counts)

# def hashingvectorize(corpus):
