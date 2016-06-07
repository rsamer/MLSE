# -*- coding: utf-8 -*-

import logging
from time import time
from sklearn.feature_extraction.text import TfidfVectorizer

_logger = logging.getLogger(__name__)


def extract_tokens(post):
    return post.tokens()


def tfidf(train_posts, test_posts, max_features=None, min_df=1, max_df=1.0, norm="l2"):
    _logger.info("TFIDF-Vectorizer (Transformation)")
    #vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
    vectorizer = TfidfVectorizer(stop_words=None,
                                 #ngram_range=(1, 3), # no impact when using our own tokenizer/preprocessor/analyzer
                                 preprocessor=extract_tokens,
                                 analyzer=extract_tokens,
                                 tokenizer=extract_tokens,
                                 #token_pattern=r'.*',
                                 min_df=min_df, # get rid of noise!
                                 max_df=max_df,
                                 use_idf=True,
                                 sublinear_tf=False,
                                 smooth_idf=True,
                                 norm=norm,
                                 max_features=max_features)

    _logger.debug("Extracting features from the training data using a sparse vectorizer")
    t0 = time()
    X_train = vectorizer.fit_transform(train_posts)
    duration = time() - t0
    _logger.debug("duration: %d" % duration)
    _logger.debug("n_samples: %d, n_features: %d" % X_train.shape)
    assert len(train_posts) == X_train.shape[0]

    _logger.debug("Extracting features from the test data using the same vectorizer")
    t0 = time()
    X_test = vectorizer.transform(test_posts)
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
    assert len(features) == len(vectorizer.vocabulary_)
    removed_features = set()
    for idx, d in enumerate(X_train):
        tokens = []
        for i in d.indices:
            tokens += [features[i]]
        for expected_token in train_posts[idx].tokens():
            #assert expected_token in tokens
            if expected_token not in tokens:
                removed_features.add(expected_token)
    assert len(vectorizer.stop_words_) == len(removed_features)
    # }
    _logger.info("Removed %d features by TfidfVectorizer", len(vectorizer.stop_words_))
    _logger.info("%d features used for training", len(vectorizer.vocabulary_))
    return X_train, X_test
