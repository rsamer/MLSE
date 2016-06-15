# -*- coding: utf-8 -*-

import logging
from time import time
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

_logger = logging.getLogger(__name__)


def tfidf(X_train, X_test, max_features=None, min_df=1, max_df=1.0, norm="l2"):
    _logger.info("TFIDF-Vectorizer (Transformation)")
    vectorizer = CountVectorizer(input='content',
                                 encoding='utf-8',
                                 decode_error='strict',
                                 strip_accents=None,
                                 lowercase=True,
                                 preprocessor=None,
                                 tokenizer=lambda text: text.split(),
                                 stop_words=None,
                                 ngram_range=(2,3),
                                 analyzer='word',
                                 max_df=max_df,
                                 min_df=min_df,
                                 max_features=max_features,
                                 vocabulary=None,
                                 binary=False)
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

    # TFIDF transformation
    transformer = TfidfTransformer(use_idf=True, sublinear_tf=False, smooth_idf=True, norm=norm)
    X_train_new = transformer.fit_transform(X_train_new)
    X_test_new = transformer.transform(X_test_new)

    ####
    # PRINT TOP N FEATURES
    print len(vectorizer.get_feature_names())
    print len(set(vectorizer.get_feature_names()))
    for f in vectorizer.get_feature_names():
        print f
    import sys;sys.exit()
    from collections import defaultdict
    features_by_gram = defaultdict(list)
    for f, w in zip(transformer.get_feature_names(), transformer.idf_):
        features_by_gram[len(f.split(' '))].append((f, w))
    top_n = 100
    for gram, features in features_by_gram.iteritems():
        top_features = sorted(features, key=lambda x: x[1], reverse=True)[:top_n]
        top_features = [f[0] for f in top_features]
        print '{}-gram top:'.format(gram), top_features
        print '-'*80
    print '-----'
    for f in transformer.get_feature_names()[-100:]:
        print f
    print '-----'
    ####



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
    return X_train_new, X_test_new
