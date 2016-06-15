# -*- coding: utf-8 -*-

import logging
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from unsupervised import kmeans, hac
import numpy as np
from util import helper
from sklearn import metrics
from sklearn.grid_search import GridSearchCV

_logger = logging.getLogger(__name__)


def extract_tokens(text):
    return text.split()


def _grid_search_cluster(model, parameters, X_train, y_train, X_test, y_test, tags, n_suggested_tags, use_numeric_features):
    if not use_numeric_features:
        parameters['vectorizer__max_features'] = (None, 2000, 10000, 20000)
        parameters['vectorizer__max_df'] = (0.8, 1.0)
        parameters['vectorizer__min_df'] = (2, 4)
        parameters['vectorizer__ngram_range'] = ((1, 1), (1, 3))  # unigrams, bigrams or trigrams
        parameters['tfidf__use_idf'] = (True,)

        pipeline = Pipeline([
            ('vectorizer', CountVectorizer(
                input='content',
                tokenizer=extract_tokens,
                preprocessor=None,
                analyzer='word',
                encoding='utf-8',
                decode_error='strict',
                strip_accents=None,
                lowercase=True,
                stop_words=None,
                vocabulary=None,
                binary=False
            )),
            ('tfidf', TfidfTransformer(
                norm='l2',
                sublinear_tf=False,
                smooth_idf=True,
                use_idf=True
            )),
            model
        ])
    else:
        pipeline = Pipeline([
            model
        ])

    pipeline = GridSearchCV(pipeline, parameters, n_jobs=-1, cv=3, verbose=0, scoring='f1_micro')

    pipeline.fit(np.array(X_train) if not use_numeric_features else X_train, y_train)

    _logger.info("Best parameters set:")
    best_parameters = pipeline.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        _logger.info("\t%s: %r" % (param_name, best_parameters[param_name]))

    y_predicted = pipeline.predict(X_test)

    print "=" * 80
    print "  REPORT FOR FIXED TAG SIZE = %d" % n_suggested_tags
    print "=" * 80

    precision_micro, recall_micro, f1_micro, _ = metrics.precision_recall_fscore_support(y_test,
                                                                                         y_predicted,
                                                                                         average="micro", warn_for=())
    precision_macro, recall_macro, f1_macro, _ = metrics.precision_recall_fscore_support(y_test,
                                                                                         y_predicted,
                                                                                         average="macro", warn_for=())

    print "Precision micro: %.3f" % precision_micro
    print "Precision macro: %.3f" % precision_macro
    print "Recall micro: %.3f" % recall_micro
    print "Recall macro: %.3f" % recall_macro
    print "F1 micro: %.3f" % f1_micro
    print "F1 macro: %.3f" % f1_macro


def clustering(X_train, y_train, X_test, y_test, tags, n_suggested_tags, use_numeric_features):
    _logger.info("-" * 80)
    _logger.info("kMeans...")

    params = {
        'kmeans__n_clusters': (len(tags), int(len(tags) / 2.0), int(len(tags) / 4.0)),
        'kmeans__max_iter': (100, ),  # max iterations per single run
        'kmeans__n_init': (10, ),  # number of times to init with different centroid seeds
        'kmeans__tol': (0.0025, 0.5, 2.0, 5.0)  # stop iteration when the change in inertia is smaller than tol (convergence criteria)
    }

    model = ('kmeans', kmeans.CustomKMeans(n_suggested_tags=n_suggested_tags,
                                           n_clusters=len(tags),
                                           init='k-means++',  # to speed up convergence
                                           n_jobs=1,  # parallel computation is handled by GridSearchCV
                                           precompute_distances='auto',  # do not precompute distances if posts * tags > 12 million (full data set = 61.842.024)
                                           copy_x=True,  # do not modify original data
                                           verbose=False))

    _grid_search_cluster(model, params, X_train, y_train, X_test, y_test, tags, n_suggested_tags, use_numeric_features)

    _logger.info("-" * 80)
    _logger.info("Agglomerative Clustering...")

    params = {
        'hac__n_clusters': (len(tags), len(tags) / 2),
        'hac__linkage': ('ward', 'complete', 'average')
    }

    model = ('hac', hac.CustomHAC(n_suggested_tags=n_suggested_tags,
                                  n_clusters=len(tags),
                                  affinity="euclidean",  # if linkage is "ward", only "euclidean" is accepted.
                                  connectivity=None,  # hierarchical clustering algorithm is unstructured (no connectivity matrix)
                                  memory=helper.HAC_CACHE_PATH,  # cache the output of the computation of the tree
                                  compute_full_tree=True))  # always compute full tree since caching is activated and the number of clusters varies (GridSearchCV)

    _grid_search_cluster(model, params, X_train, y_train, X_test, y_test, tags, n_suggested_tags, use_numeric_features)
