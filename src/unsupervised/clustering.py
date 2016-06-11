# -*- coding: utf-8 -*-

import logging
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from unsupervised import kmeans
import numpy as np
from sklearn.grid_search import GridSearchCV

_logger = logging.getLogger(__name__)


def cluster(X_train, y_train, X_test, y_test, tags, n_suggested_tags, use_numeric_features):
    parameters = {
        'kmeans__n_clusters': (len(tags), int(len(tags) / 2)),
        'kmeans__max_iter': (20, 100),
        'kmeans__n_init': (10, 100),
        'kmeans__tol': (0.00004, 0.001)
    }

    if not use_numeric_features:
        parameters['vectorizer__max_features'] = (None, 1000, 3000, 10000)
        parameters['vectorizer__max_df'] = (0.75, 0.8, 1.0)
        parameters['vectorizer__min_df'] = (2, 3)
        parameters['vectorizer__ngram_range'] = ((1, 1), (1, 2), (1, 3))  # unigrams, bigrams or trigrams
        parameters['tfidf__use_idf'] = (True, False)

        pipeline = Pipeline([
            ('vectorizer', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('kmeans', kmeans.CustomKMeans(n_clusters=len(tags), init='k-means++', verbose=True, n_jobs=-1))
        ])
    else:
        pipeline = Pipeline([
            ('kmeans', kmeans.CustomKMeans(init='k-means++', verbose=True, n_jobs=-1))
        ])

    pipeline = GridSearchCV(pipeline, parameters, n_jobs=-1, cv=3, verbose=0)

    pipeline.fit(np.array(X_train) if not use_numeric_features else X_train, y_train)

    _logger.info("Best parameters set:")
    best_parameters = pipeline.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        _logger.info("\t%s: %r" % (param_name, best_parameters[param_name]))

    y_predicted = pipeline.predict(np.array(X_test) if not use_numeric_features else X_test, n_suggested_tags=n_suggested_tags)

    print "=" * 80
    print "  REPORT FOR FIXED TAG SIZE = %d" % n_suggested_tags
    print "=" * 80
    from sklearn import metrics

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
