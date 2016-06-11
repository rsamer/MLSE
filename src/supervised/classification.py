# -*- coding: utf-8 -*-

import logging
from sklearn import metrics
from time import time
from util import helper

_logger = logging.getLogger(__name__)


def train_and_test_classifier_for_single_tag(classifier, tag_name, X_train, y_train, X_test, y_test):
    _logger.debug("Training: %s" % tag_name)
    t0 = time()
    classifier.fit(X_train, y_train)
    train_time = time() - t0
    t0 = time()
    #if hasattr(classifier, "decision_function"):
    #prediction_list = classifier.predict(X_test)
    #    prediction_probabilities_list = classifier.decision_function(X_test) # for ensemble methods!
    #else:
    prediction_probabilities_list = classifier.predict_proba(X_test)

    # sanity checks!
    classes = classifier.classes_
    assert len(classes) == 1 or len(classes) == 2
    assert classes[0] == False
    if len(classes) == 2:
        assert classes[1] == True

    for p1, p2 in prediction_probabilities_list:
        assert abs(1.0 - (p1+p2)) < 0.001

    prediction_positive_probabilities = map(lambda p: p[1], prediction_probabilities_list)
    prediction_list = map(lambda p: p[1] > p[0], prediction_probabilities_list)
    test_time = time() - t0
    score = metrics.accuracy_score(y_test, prediction_list)
    return tag_name, prediction_positive_probabilities, score, train_time, test_time


def one_vs_rest(clf, X_train, y_train):
    from sklearn.multiclass import OneVsRestClassifier
    _logger.info("%s - OneVsRestClassifier", clf.__class__.__name__)
    one_vs_rest_clf = OneVsRestClassifier(clf, n_jobs=1)#-1)
    t0 = time()
    one_vs_rest_clf.fit(X_train, y_train)
    train_time = time() - t0
    return one_vs_rest_clf, train_time


def classification(classifier, X_train, X_test, train_posts, test_posts, tags):
    _logger.info("%s - One classifier per tag", classifier.__class__.__name__)
    progress_bar = helper.ProgressBar(len(tags))
    test_post_tag_prediction_map = {}
    results = []
    for tag in tags:
        tag_name = tag.name
        y_train = map(lambda p: tag_name in map(lambda t: t.name, p.tag_set), train_posts)
        y_test = map(lambda p: tag_name in map(lambda t: t.name, p.tag_set), test_posts)
        result = train_and_test_classifier_for_single_tag(classifier, tag_name, X_train, y_train, X_test, y_test)
        results.append(result)
        prediction_positive_probabilities_of_posts = result[1]
        for idx, test_post in enumerate(test_posts):
            positive_probability_of_post = prediction_positive_probabilities_of_posts[idx]
            if test_post not in test_post_tag_prediction_map:
                test_post_tag_prediction_map[test_post] = []
            test_post_tag_prediction_map[test_post] += [(tag, positive_probability_of_post)]
        progress_bar.update()
    progress_bar.finish()

    avg_score = float(reduce(lambda x,y: x+y, map(lambda r: r[2], results)))/float(len(results))
    total_train_time = reduce(lambda x,y: x+y, map(lambda r: r[3], results))
    total_test_time = reduce(lambda x,y: x+y, map(lambda r: r[4], results))
    _logger.info("Total train time: %0.3fs", total_train_time)
    _logger.info("Total test time: %0.3fs", total_test_time)
    _logger.info("Average score: %0.3f%%", avg_score*100.0)

    for idx, test_post in enumerate(test_posts):
        if test_post not in test_post_tag_prediction_map:
            test_post.tag_set_prediction = []
            continue

        sorted_tag_predictions = sorted(test_post_tag_prediction_map[test_post], key=lambda p: p[1], reverse=True)
        sorted_tags = map(lambda p: p[0], sorted_tag_predictions)
        test_post.tag_set_prediction = sorted_tags[:2]
        _logger.debug("Suggested Tags for test-post = {}{}".format(test_post, sorted_tags[:10]))

