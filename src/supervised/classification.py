# -*- coding: utf-8 -*-

import logging
from time import time
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import precision_recall_fscore_support
from evaluation.classification import custom_classification_report

_logger = logging.getLogger(__name__)


def _grid_search_classification(model, parameters, X_train, y_train, X_test, y_test, mlb, tags,
                                n_suggested_tags, use_numeric_features):
    #-----------------------------------------------------------------------------------------------
    # SETUP PARAMETERS & CLASSIFIERS
    #-----------------------------------------------------------------------------------------------
    if not use_numeric_features:
        parameters['vectorizer__max_features'] = (None, 1000, 2000, 3000)
        parameters['vectorizer__max_df'] = (0.85, 1.0)
        parameters['vectorizer__min_df'] = (2, 4)
        parameters['vectorizer__ngram_range'] = ((1, 1), (1, 2), (1, 3))  # unigrams, bigrams or trigrams
        parameters['tfidf__use_idf'] = (True, )

        classifier = Pipeline([
            ('vectorizer', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            model
        ])
    else:
        classifier = Pipeline([model])

    #-----------------------------------------------------------------------------------------------
    # LEARNING / FIT CLASSIFIER / GRIDSEARCH VIA CROSS VALIDATION (OF TRAINING DATA)
    #-----------------------------------------------------------------------------------------------
    classifier = GridSearchCV(classifier, parameters, n_jobs=-1, cv=3, verbose=1, scoring='f1_micro')

    _logger.info("Parameters: %s", parameters)
    t0 = time()
    classifier.fit(np.array(X_train) if not use_numeric_features else X_train, y_train)
    _logger.info("Done in %0.3fs" % (time() - t0))

    _logger.info("Best parameters set:")
    best_parameters = classifier.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        _logger.info("\t%s: %r" % (param_name, best_parameters[param_name]))

    #-----------------------------------------------------------------------------------------------
    # PREDICTION
    #-----------------------------------------------------------------------------------------------
    _logger.info("Number of suggested tags: %d" % n_suggested_tags)
    y_predicted = classifier.predict(np.array(X_test) if not use_numeric_features else X_test)
    y_predicted_probab = classifier.predict_proba(np.array(X_test) if not use_numeric_features else X_test)
    y_predicted_list = []
    y_predicted_label_list = []
    tag_names_in_labelized_order = list(mlb.classes_)
    for probabilities in y_predicted_probab:
        top_tag_predictions = sorted(enumerate(probabilities), key=lambda p: p[1], reverse=True)[:n_suggested_tags]
        top_tag_prediction_indexes = map(lambda (idx, _): idx, top_tag_predictions)
        y_predicted_list.append(map(lambda i: int(i in top_tag_prediction_indexes), range(len(tag_names_in_labelized_order))))
        predicted_tag_names = map(lambda idx: tag_names_in_labelized_order[idx], top_tag_prediction_indexes)
        y_predicted_label_list.append(predicted_tag_names)

    y_predicted_fixed_size = np.array(y_predicted_list)

    # sanity check to ensure the code in the for-loop above is doing right thing!
    for idx, predicted_tag_names_for_post in enumerate(mlb.inverse_transform(y_predicted_fixed_size)):
        assert set(predicted_tag_names_for_post) == set(y_predicted_label_list[idx])

    #-----------------------------------------------------------------------------------------------
    # EVALUATION
    #-----------------------------------------------------------------------------------------------

    #-----------------------------------------------------------------------------------------------
    # NOTE: uncomment this if you want variable tag size
    #-----------------------------------------------------------------------------------------------
#     print "-"*80
#     for item, labels in zip(X_test, mlb.inverse_transform(y_predicted)):
#         print '%s -> (%s)' % (item[:40], ', '.join(labels))
#
#     print "="*80
#     print "  REPORT FOR VARIABLE TAG SIZE"
#     print "="*80
#     print classification_report(y_test_mlb, y_predicted)
    #-----------------------------------------------------------------------------------------------

    if not use_numeric_features:
        print "-"*80
        for item, labels in zip(X_test, y_predicted_label_list):
            print '%s -> (%s)' % (item[:40], ', '.join(labels))

    print "="*80
    print "  REPORT FOR FIXED TAG SIZE = %d" % n_suggested_tags
    print "="*80

    p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(y_test, y_predicted_fixed_size,
                                                                    average="micro", warn_for=())
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_test, y_predicted_fixed_size,
                                                                    average="macro", warn_for=())

    print custom_classification_report(y_test, y_predicted_fixed_size, target_names=list(mlb.classes_))
    print "Precision micro: %.3f" % p_micro
    print "Precision macro: %.3f" % p_macro
    print "Recall micro: %.3f" % r_micro
    print "Recall macro: %.3f" % r_macro
    print "F1 micro: %.3f" % f1_micro
    print "F1 macro: %.3f" % f1_macro


def classification(X_train, y_train, X_test, y_test, mlb, tags, n_suggested_tags, use_numeric_features):

    parameters = {
#         'clf__alpha': [0.2, 0.1, 0.06, 0.03, 0.01, 0.001, 0.0001],
    }

    models = [
        # baseline
#        ('clf', OneVsRestClassifier(DummyClassifier("most_frequent"))), # very primitive/simple baseline!
        ('clf', OneVsRestClassifier(MultinomialNB(alpha=.03))), # <-- lidstone smoothing (1.0 would be laplace smoothing!)

#         # "single" classifiers
        ('clf', OneVsRestClassifier(KNeighborsClassifier(n_neighbors=10))),
        ('clf', OneVsRestClassifier(SVC(kernel="linear", C=0.025, probability=True))),
#         ('clf', OneVsRestClassifier(SVC(kernel="linear", C=2.0, probability=True))),
#         ('clf', OneVsRestClassifier(SVC(kernel="rbf", C=0.025, probability=True))),
#         #('clf', OneVsRestClassifier(LinearSVC())),

        # ensemble
#        ('clf', OneVsRestClassifier(RandomForestClassifier(n_estimators=200, max_depth=None))),
#        ('clf', OneVsRestClassifier(AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), n_estimators=200, learning_rate=1.5, algorithm="SAMME")))
    ]

    for model in models:
        _logger.info(str(model[1]))
        _grid_search_classification(model, parameters, X_train, y_train, X_test, y_test, mlb, tags,
                                    n_suggested_tags, use_numeric_features)


#---------------------------------------------------------------------------------------------------
# legacy code:
# for clf in classifiers:
#     print y_train_mlb
#     one_vs_rest_clf, _ = classification.one_vs_rest(clf, np.array(X_train), y_train_mlb)
#     y_pred_mlb = one_vs_rest_clf.predict(np.array(X_test))
#     print classification_report(mlb.transform(y_test), y_pred_mlb)
#     classification.classification(classifier, X_train, X_test, train_posts, test_posts, tags)
#     evaluation.print_evaluation_results(test_posts)
# 
#     # sanity checks!
#     assert classifier.classes_[0] == False
#     assert classifier.classes_[1] == True
#     for p1, p2 in prediction_probabilities_list:
#         assert abs(1.0 - (p1+p2)) < 0.001
#
# Suggest most frequent tags
# Random Classifier
#     _logger.info("-"*80)
#     _logger.info("Randomly suggest 2 most frequent tags...")
#     helper.suggest_random_tags(2, test_posts, tags)
#     evaluation.print_evaluation_results(test_posts)
#     _logger.info("-"*80)
#     _logger.info("Only auggest most frequent tag '%s'..." % tags[0])
#     helper.suggest_random_tags(1, test_posts, [tags[0]])
#     evaluation.print_evaluation_results(test_posts)


'''
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
'''
