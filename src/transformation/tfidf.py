# -*- coding: utf-8 -*-

import logging
from time import time
from sklearn.feature_extraction.text import TfidfVectorizer

_logger = logging.getLogger(__name__)

def tfidf(train_posts, test_posts):
    train_documents = [" ".join(post.tokens) for post in train_posts]
    test_documents = [" ".join(post.tokens) for post in test_posts]

    _logger.info("TFIDF-Vectorizer (Transformation)")
    #vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
    vectorizer = TfidfVectorizer(stop_words=None) # max_features=2000

    _logger.debug("Extracting features from the training data using a sparse vectorizer")
    t0 = time()
    X_train = vectorizer.fit_transform(train_documents)
    duration = time() - t0
    _logger.debug("duration: %d" % duration)
    _logger.debug("n_samples: %d, n_features: %d" % X_train.shape)
    assert len(train_posts) == X_train.shape[0]

    _logger.debug("Extracting features from the test data using the same vectorizer")
    t0 = time()
    X_test = vectorizer.transform(test_documents)
    duration = time() - t0
    _logger.debug("duration: %d" % duration)
    _logger.debug("n_samples: %d, n_features: %d" % X_test.shape)

    _logger.info("X_train: {}".format(X_train.shape))
    _logger.info("X_test: {}".format(X_test.shape))

    assert len(test_posts) == X_test.shape[0]
    assert len(test_posts) == X_test.shape[0]
    assert X_train.shape[1] == X_test.shape[1]

    # SANITY CHECK {
    features = vectorizer.get_feature_names()
    critical_token_names = set()
    for idx, d in enumerate(X_train):
        tokens = []
        for i in d.indices:
            tokens += [features[i]]
        for expected_token in train_documents[idx].split():
            #assert expected_token in tokens
            if expected_token not in tokens:
                critical_token_names.add(expected_token)
    for idx, d in enumerate(X_test):
        tokens = []
        for i in d.indices:
            tokens += [features[i]]
        for expected_token in test_documents[idx].split():
            #assert expected_token in tokens
            if expected_token not in tokens:
                critical_token_names.add(expected_token)
    print "-"*80
    print "Critical token names:"
    print critical_token_names
    print "-"*80
    print "-"*80
    print "-"*80
    print features[:200]
#     import sys;sys.exit()
    # }

    return X_train, X_test
