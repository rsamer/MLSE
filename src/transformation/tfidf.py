# -*- coding: utf-8 -*-

import logging
from time import time
from sklearn.feature_extraction.text import TfidfVectorizer

_logger = logging.getLogger(__name__)


def extract_tokens(post):
    return post.tokens(title_weight=3)


#     _logger.info("-" * 80)
#     _logger.info("Transformation...")
#     n_features = 20000 #2500  # 2200 # 2500 for KNN
#     from transformation import tfidf, features
#     X_train, X_test = tfidf.tfidf(train_posts, test_posts, max_features=None, min_df=2)
#X_train, X_test = features.numeric_features(train_posts, test_posts, tags)

def tfidf(X_train, X_test, max_features=None, min_df=1, max_df=1.0, norm="l2"):
    _logger.info("TFIDF-Vectorizer (Transformation)")
    #vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
    vectorizer = TfidfVectorizer(stop_words=None,
                                 #ngram_range=(1, 3), # no impact when using our own tokenizer/preprocessor/analyzer
#                                 preprocessor=extract_tokens,
#                                 analyzer=extract_tokens,
#                                 tokenizer=extract_tokens,
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
    X_train_new = vectorizer.fit_transform(X_train)
    duration = time() - t0
    _logger.debug("duration: %d" % duration)
    _logger.debug("n_samples: %d, n_features: %d" % X_train_new.shape)
    assert len(X_train) == X_train_new.shape[0]

    _logger.debug("Extracting features from the test data using the same vectorizer")
    t0 = time()
    X_test_new = vectorizer.transform(X_test)
    duration = time() - t0
    _logger.debug("duration: %d" % duration)
    _logger.debug("n_samples: %d, n_features: %d" % X_test_new.shape)

    _logger.info("X_train: {}".format(X_train_new.shape))
    _logger.info("X_test: {}".format(X_test_new.shape))

    assert len(X_test) == X_test_new.shape[0]
    assert len(X_test) == X_test_new.shape[0]
    assert X_train_new.shape[1] == X_test_new.shape[1]

    # SANITY CHECK {
    features = vectorizer.get_feature_names()
    assert len(features) == len(vectorizer.vocabulary_)
#     removed_features = set()
#     for idx, d in enumerate(X_train):
#         tokens = []
#         for i in d.indices:
#             tokens += [features[i]]
#         for expected_token in train_posts[idx].tokens(title_weight=3):
#             #assert expected_token in tokens
#             if expected_token not in tokens:
#                 removed_features.add(expected_token)
#    assert len(vectorizer.stop_words_) == len(removed_features)
    # }
    print vectorizer.stop_words_
    _logger.info("Removed %d features by TfidfVectorizer", len(vectorizer.stop_words_))
    _logger.info("%d features used for training", len(vectorizer.vocabulary_))
    import sys;sys.exit()
    return X_train_new, X_test_new
