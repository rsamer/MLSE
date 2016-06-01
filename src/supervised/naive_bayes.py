# -*- coding: utf-8 -*-

import logging
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.utils.extmath import density
from time import time
from util import helper
from transformation import tfidf

_logger = logging.getLogger(__name__)

def train_and_test_bayes_for_single_tag(tag_name, X_train, y_train, X_test, y_test):
    _logger.debug("Training: %s" % tag_name)
    #nb_classifier = KNeighborsClassifier(n_neighbors=10) # f1 = 0.386 (features=2900)
    nb_classifier = RandomForestClassifier(n_estimators=100, max_depth=None, n_jobs=-1) # f1=0.390
    #nb_classifier = SVC(kernel="linear", C=0.025, probability=True)
    #nb_classifier = SVC(kernel="rbf", C=0.025, probability=True)
    #penalty = "l2" #"l1"
    #nb_classifier = LinearSVC(loss='l2', penalty=penalty, dual=False, tol=1e-3)
    #nb_classifier = MultinomialNB(alpha=.03) # <-- lidstone smoothing (1.0 would be laplace smoothing!)
    #nb_classifier = BernoulliNB(alpha=.01)
    t0 = time()
    nb_classifier.fit(X_train, y_train)
    train_time = time() - t0

    t0 = time()
    #prediction_list = nb_classifier.predict(X_test)
    prediction_probabilities_list = nb_classifier.predict_proba(X_test)

    # sanity checks!
    classes = nb_classifier.classes_
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


def train_and_test_bayes_for_all_tags(X_train, y_train, X_test, y_test):
    from sklearn.multiclass import OneVsRestClassifier
    print "Training:"
#     X_train = [[0, 0], [0, 1], [1, 1]]
#     y_train = [('first',), ('second',), ('first', 'second')]
    nb_classifier = OneVsRestClassifier(KNeighborsClassifier(n_neighbors=10), n_jobs=-1)#BernoulliNB(alpha=.01))#MultinomialNB(alpha=.01))
    t0 = time()
    nb_classifier.fit(X_train, y_train)
    print nb_classifier.classes_
    train_time = time() - t0

#     nb_classifier = MultinomialNB(alpha=.01)
#     t0 = time()
#     nb_classifier.fit(X_train, y_train)
#     train_time = time() - t0

    t0 = time()
    #prediction_list = nb_classifier.predict(X_test)
    prediction_probabilities_list = nb_classifier.predict_proba(X_test)

    # sanity checks!
#     assert nb_classifier.classes_[0] == False
#     assert nb_classifier.classes_[1] == True
#     for p1, p2 in prediction_probabilities_list:
#         assert abs(1.0 - (p1+p2)) < 0.001

    test_time = time() - t0
    return prediction_probabilities_list, nb_classifier.classes_, train_time, test_time


def naive_bayes_single_classifier(train_posts, test_posts, tags):
    _logger.info("Naive Bayes - Single classifier")
    X_train, X_test = tfidf.tfidf(train_posts, test_posts)

    print("Naive Bayes")
    #all_tag_names = map(lambda t: t.name, tags)
    y_train = map(lambda p: tuple(map(lambda t: t.name, p.tag_set)), train_posts)
    y_test = map(lambda p: tuple(map(lambda t: t.name, p.tag_set)), test_posts)
    #y_test = map(lambda p: tag_name in map(lambda t: t.name, p.tag_set), test_posts)
    result = train_and_test_bayes_for_all_tags(X_train, y_train, X_test, y_test)
    prediction_positive_probabilities_of_posts = result[0]
    tag_name_map = {}
    for tag in tags:
        tag_name_map[tag.name] = tag
    tag_names = result[1]
    print prediction_positive_probabilities_of_posts
    for post_idx, post_probabilities in enumerate(prediction_positive_probabilities_of_posts):
        post_probabilities_map = [(tag_name_map[tag_names[tag_idx]], p) for tag_idx, p in enumerate(post_probabilities)]
        sorted_tag_predictions = sorted(post_probabilities_map, key=lambda p: p[1], reverse=True)
        sorted_tags = map(lambda p: p[0], sorted_tag_predictions)
        test_posts[post_idx].tag_set_prediction = sorted_tags[:2]
    return
#    test_post_tag_prediction_map = {}
#     for idx, test_post in enumerate(test_posts):
#         positive_probability_of_post = prediction_positive_probabilities_of_posts[idx]
#         if test_post not in test_post_tag_prediction_map:
#             test_post_tag_prediction_map[test_post] = []
#         test_post_tag_prediction_map[test_post] += [(tag, positive_probability_of_post)]


def naive_bayes(X_train, X_test, train_posts, test_posts, tags):
    _logger.info("Naive Bayes - One classifier per tag")
    progress_bar = helper.ProgressBar(len(tags))
    test_post_tag_prediction_map = {}
    results = []
    for tag in tags:
        tag_name = tag.name
        y_train = map(lambda p: tag_name in map(lambda t: t.name, p.tag_set), train_posts)
        y_test = map(lambda p: tag_name in map(lambda t: t.name, p.tag_set), test_posts)
        result = train_and_test_bayes_for_single_tag(tag_name, X_train, y_train, X_test, y_test)
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

        # TODO: not sure if a higher score will actually be a better match...
        sorted_tag_predictions = sorted(test_post_tag_prediction_map[test_post], key=lambda p: p[1], reverse=True)
        sorted_tags = map(lambda p: p[0], sorted_tag_predictions)
        test_post.tag_set_prediction = sorted_tags[:2]
        _logger.debug("Suggested Tags for test-post = {}{}".format(test_post, sorted_tags[:10]))

    return



###############################################################################
# Benchmark classifiers
def benchmark(classifier, X_train, y_train, X_test, y_test):
    print('_' * 80)
    print("Training: ")
    print(classifier)
    t0 = time()
    classifier.fit(X_train, y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time()
    pred = classifier.predict(X_test)
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)

    score = metrics.accuracy_score(y_test, pred)
    print("accuracy:   %0.3f" % score)

    if hasattr(classifier, 'coef_'):
        print("dimensionality: %d" % classifier.coef_.shape[1])
        print("density: %f" % density(classifier.coef_))
        print()

    print("classification report:")
    print(metrics.classification_report(y_test, pred, target_names=None))#categories))
    print("confusion matrix:")
    print(metrics.confusion_matrix(y_test, pred))
    print()
    clf_descr = str(classifier).split('(')[0]
    return clf_descr, score, train_time, test_time


def plot_results(results):
    # make some plots
    indices = np.arange(len(results))
    results = [[x[i] for x in results] for i in range(4)]
    clf_names, score, training_time, test_time = results
    training_time = np.array(training_time) / np.max(training_time)
    test_time = np.array(test_time) / np.max(test_time)

    plt.figure(figsize=(12, 8))
    plt.title("Score")
    plt.barh(indices, score, .2, label="score", color='r')
    plt.barh(indices + .3, training_time, .2, label="training time", color='g')
    plt.barh(indices + .6, test_time, .2, label="test time", color='b')
    plt.yticks(())
    plt.legend(loc='best')
    plt.subplots_adjust(left=.25)
    plt.subplots_adjust(top=.95)
    plt.subplots_adjust(bottom=.05)
    for i, c in zip(indices, clf_names):
        plt.text(-.3, i, c)
    plt.show()

    #################


#     for clf, name in (
#             (RidgeClassifier(tol=1e-2, solver="lsqr"), "Ridge Classifier"),
#             (Perceptron(n_iter=50), "Perceptron"),
#             (PassiveAggressiveClassifier(n_iter=50), "Passive-Aggressive"),
#             (KNeighborsClassifier(n_neighbors=10), "kNN"),
#             (RandomForestClassifier(n_estimators=100), "Random forest")):
#         print('=' * 80)
#         print(name)
#         results.append(benchmark(clf, X_train, y_train, X_test, y_test))
#
#     for penalty in ["l2", "l1"]:
#         print('=' * 80)
#         print("%s penalty" % penalty.upper())
#         # Train Liblinear model
#         results.append(benchmark(LinearSVC(loss='l2', penalty=penalty,
#                                                 dual=False, tol=1e-3),
#                                  X_train, y_train, X_test, y_test))
#         # Train SGD model
#         results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50,
#                                                penalty=penalty),
#                                  X_train, y_train, X_test, y_test))
# 
#     # Train SGD with Elastic Net penalty
#     print('=' * 80)
#     print("Elastic-Net penalty")
#     results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50,
#                                            penalty="elasticnet"),
#                              X_train, y_train, X_test, y_test))
# 
#     # Train NearestCentroid without threshold
#     print('=' * 80)
#     print("NearestCentroid (aka Rocchio classifier)")
#     results.append(benchmark(NearestCentroid(), X_train, y_train, X_test, y_test))
#
#     # Train sparse Naive Bayes classifiers
#     print('=' * 80)
#     print("Naive Bayes")
#     results.append(benchmark(MultinomialNB(alpha=.01), X_train, y_train, X_test, y_test))
#     results.append(benchmark(BernoulliNB(alpha=.01), X_train, y_train, X_test, y_test))
# 
#     print('=' * 80)
#     print("LinearSVC with L1-based feature selection")
#     # The smaller C, the stronger the regularization.
#     # The more regularization, the more sparsity.
#     results.append(benchmark(Pipeline([
#       ('feature_selection', LinearSVC(penalty="l1", dual=False, tol=1e-3)),
#       ('classification', LinearSVC())
#     ]), X_train, y_train, X_test, y_test))
