#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
    main -- Main entry point
    
    main is a function in this module that performs all steps of the entire pipeline
    (preprocessing, training, testing and evaluation)
    
    Note: You may have to install the following libraries first:
    * numpy
    * matplotlib
    TODO: update this list!
    
    usage:
    python -m main ../data/example/
    
    -----------------------------------------------------------------------------
    NOTE: the data set located in ../data/example/ is NOT a representative subset
          of the entire "programmers.stackexchange.com" data set
    -----------------------------------------------------------------------------
    
    @author:     Michael Herold, Ralph Samer
    @copyright:  2016 organization_name. All rights reserved.
    @license:    MIT license
'''

import logging
import sys
from time import time
import numpy as np
from entities.tag import Tag
from preprocessing import parser, preprocessing as prepr
from sklearn.cross_validation import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
# from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn import metrics
# from unsupervised import kmeans, hac
from transformation import features
from util import helper
from util.docopt import docopt
from util.helper import ExitCode

__version__ = 1.0
_logger = logging.getLogger(__name__)


def usage():
    DEFAULT_TAG_FREQUENCY_THRESHOLD = 3
    DEFAULT_N_SUGGESTED_TAGS = 2
    DEFAULT_TEST_SIZE = 0.1
    DEFAULT_USE_NUMERIC_FEATURES = False
    usage = '''
        Automatic Tag Suggestion for StackExchange posts
    
        Usage:
          'main.py' <data-set-path> [--use-caching] [--use-numeric-features] [--num-suggested-tags=<nt>] [--tag-frequ-thr=<tf>] [--test-size=<ts>]
          'main.py' --version
    
        Options:
          -h --help                         Shows this screen.
          -v --version                      Shows version of this application.
          -c --use-caching                  Enables caching in order to avoid redundant preprocessing.
          -n --use-numeric-features         Enables numeric features (PMI) instead of TFxIDF
          -s=<nt> --num-suggested-tags=<nt> Number of suggested tags (default=%d)
          -f=<tf> --tag-frequ-thr=<tf>      Sets tag frequency threshold -> appropriate value depends on which data set is used! (default=%d)
          -t=<ts> --test-size=<ts>          Sets test size (range: 0.01-0.5) (default=%f)
    ''' % (DEFAULT_N_SUGGESTED_TAGS, DEFAULT_TAG_FREQUENCY_THRESHOLD, DEFAULT_TEST_SIZE)
    arguments = docopt(usage)
    kwargs = {}
    kwargs['enable_caching'] = arguments["--use-caching"]
    kwargs['numeric_features'] = arguments["--use-numeric-features"] if arguments["--use-numeric-features"] else DEFAULT_USE_NUMERIC_FEATURES
    kwargs['n_suggested_tags'] = int(arguments["--num-suggested-tags"]) if arguments["--num-suggested-tags"] else DEFAULT_N_SUGGESTED_TAGS
    kwargs['tag_frequency_threshold'] = int(arguments["--tag-frequ-thr"]) if arguments["--tag-frequ-thr"] else DEFAULT_TAG_FREQUENCY_THRESHOLD
    kwargs['test_size'] = max(0.01, min(0.5, float(arguments["--test-size"][0]))) if arguments["--test-size"] else DEFAULT_TEST_SIZE
    kwargs['data_set_path'] = arguments["<data-set-path>"]
    show_version_only = arguments["--version"]
    if show_version_only:
        print("Automatic Tag Suggestion for StackExchange posts\nVersion: {}".format(__version__))
        sys.exit(ExitCode.SUCCESS)
    # TODO: introduce CLI arguments for selecting desired pipeline and determine which algorithms to use...
    return kwargs


def setup_logging(log_level):
    import platform
    logging.basicConfig(
        filename=None, #os.path.join(helper.LOG_PATH, "automatic_tagger.log"),
        level=log_level,
        format='%(asctime)s: %(levelname)7s: [%(name)s]: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    # based on: http://stackoverflow.com/a/1336640
    if platform.system() == 'Windows':
        logging.StreamHandler.emit = helper.add_coloring_to_emit_windows(logging.StreamHandler.emit)
    else:
        logging.StreamHandler.emit = helper.add_coloring_to_emit_ansi(logging.StreamHandler.emit)


def preprocess_tags_and_posts(all_tags, all_posts, tag_frequency_threshold):
    from preprocessing import tags
    # FIXME: figure out why this fails on academia dataset!
    #filtered_tags = all_tags # f1=0.349
    # f1=0.368, f1=0.352
    filtered_tags, all_posts = tags.replace_tag_synonyms(all_tags, all_posts)
    filtered_tags = prepr.filter_tags_and_sort_by_frequency(filtered_tags, tag_frequency_threshold)
    prepr.preprocess_tags(filtered_tags)
    posts = prepr.preprocess_posts(all_posts, filtered_tags, filter_posts=True)
    Tag.update_tag_counts_according_to_posts(filtered_tags, posts)
    return filtered_tags, posts


def main():
    kwargs = usage()
    data_set_path = kwargs['data_set_path']
    enable_caching = kwargs['enable_caching']
    use_numeric_features = kwargs['numeric_features']
    n_suggested_tags = kwargs['n_suggested_tags']
    setup_logging(logging.INFO)
    helper.make_dir_if_not_exists(helper.CACHE_PATH)
    if enable_caching:
        _logger.info("Caching enabled!")

    # 1) Parsing
    _logger.info("Parsing...")
    all_tags, all_posts, cache_file_name_prefix = parser.parse_tags_and_posts(data_set_path)
    all_posts_assignments = reduce(lambda x,y: x + y, map(lambda p: len(p.tag_set), all_posts))
    tag_frequency_threshold = kwargs['tag_frequency_threshold']
    cache_file_name_prefix += "_%d_" % tag_frequency_threshold # caching depends on tag-frequency

    # 2) Preprocessing
    _logger.info("Preprocessing...")
    if not enable_caching or not helper.cache_exists_for_preprocessed_tags_and_posts(cache_file_name_prefix):
        tags, posts = preprocess_tags_and_posts(all_tags, all_posts, tag_frequency_threshold)
        helper.write_preprocessed_tags_and_posts_to_cache(cache_file_name_prefix, tags, posts)
    else:
        _logger.info("Cache hit!")
        tags, posts = helper.load_preprocessed_tags_and_posts_from_cache(cache_file_name_prefix)

    helper.print_tags_summary(len(all_tags), len(tags))
    helper.print_posts_summary(all_posts, all_posts_assignments, posts)

    # 3) Split data set
    test_size = kwargs["test_size"]
    _logger.info("Splitting data set -> Training: {}%, Test: {}%".format((1-test_size)*100, test_size*100))
    # NOTE: last 2 return values are omitted since y-values are already
    #       included in our Post-instances, therefore the 2nd argument passed
    #       to the function is irrelevant (list of zeros!)

    y = map(lambda p: tuple(map(lambda t: t.name, p.tag_set)), posts)
    if not use_numeric_features:
        X = map(lambda p: ' '.join(p.tokens()), posts)
        assert len(X) == len(y)
    else:
        X, _ = features.numeric_features(posts, [], tags, True)
        assert X.shape[0] == len(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    mlb = MultiLabelBinarizer()
    y_train_mlb = mlb.fit_transform(np.array(y_train))
    y_test_mlb = mlb.transform(np.array(y_test))

    # 3) vectorization, transformation and learning
    _logger.info("Learning...")

    # supervised
    _logger.info("-"*80)
    _logger.info("Supervised - Classification...")

    parameters = {
        #'estimator__alpha': [0.2, 0.1, 0.06, 0.03, 0.01, 0.001, 0.0001],
    }

    # TODO: remove single_classifier in classification.py -> http://stackoverflow.com/a/31586026
    if not use_numeric_features:
        classifier = Pipeline([
            ('vectorizer', CountVectorizer()),#min_df=2, max_features=None, ngram_range=(1, 3))),
            ('tfidf', TfidfTransformer()),
            #('clf', OneVsRestClassifier(SVC(kernel="linear", probability=True)))])
            #('clf', OneVsRestClassifier(LinearSVC()))])
            ('clf', OneVsRestClassifier(SVC(kernel="linear", C=0.025, probability=True)))
            #('clf', OneVsRestClassifier(MultinomialNB(alpha=.03)))
#             DummyClassifier("most_frequent"), # very primitive/simple baseline!
#             AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), n_estimators=200, learning_rate=1.5, algorithm="SAMME"),
#             SVC(kernel="linear", C=0.4, probability=True),
#             KNeighborsClassifier(n_neighbors=10), # f1 = 0.386 (features=2900)
#             RandomForestClassifier(n_estimators=200, max_depth=None, n_jobs=-1),  # f1=0.390
#             MultinomialNB(alpha=.03), # <-- lidstone smoothing (1.0 would be laplace smoothing!)
#             BernoulliNB(alpha=.01),
#             SVC(kernel="linear", C=0.025, probability=True),
#             SVC(kernel="rbf", C=0.025, probability=True), # penalty = "l2" #"l1"
#             LinearSVC(loss='l2', penalty="l2", dual=False, tol=1e-3),
        ])

        parameters['vectorizer__max_features'] = (None, 1000, 5000)
        parameters['vectorizer__max_df'] = (0.8, 1.0)
        parameters['vectorizer__min_df'] = (2, 4)
        parameters['vectorizer__ngram_range'] = ((1, 1), (1, 2), (1, 3))  # unigrams, bigrams or trigrams
        parameters['tfidf__use_idf'] = (True)
    else:
        classifier = Pipeline([
            ('clf', OneVsRestClassifier(SVC(kernel="linear", C=2.0, probability=True)))
        ])

#     for score in ['precision', 'recall']:
#     _logger.info("# Tuning hyper-parameters for %s", score)
    classifier = GridSearchCV(classifier, parameters, n_jobs=-1, cv=3, verbose=0)#, scoring='%s_weighted' % score)
    _logger.info("Parameters: %s", parameters)
    t0 = time()
    classifier.fit(np.array(X_train) if not use_numeric_features else X_train, y_train_mlb)
    _logger.info("Done in %0.3fs" % (time() - t0))
    _logger.info("Best parameters set:")
    _logger.info("Best score: %0.3f", classifier.best_score_)
    best_parameters = classifier.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        _logger.info("\t%s: %r" % (param_name, best_parameters[param_name]))

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
    # sanity check!
    for idx, predicted_tag_names_for_post in enumerate(mlb.inverse_transform(y_predicted_fixed_size)):
        assert set(predicted_tag_names_for_post) == set(y_predicted_label_list[idx])

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

#     print ""
#     print ""
#     print "-"*80
#     for item, labels in zip(X_test, mlb.inverse_transform(y_predicted)):
#         print '%s -> (%s)' % (item[:40], ', '.join(labels))

    if not use_numeric_features:
        print ""
        print ""
        print "-"*80
        for item, labels in zip(X_test, y_predicted_label_list):
            print '%s -> (%s)' % (item[:40], ', '.join(labels))

#     print "="*80
#     print "  REPORT FOR VARIABLE TAG SIZE"
#     print "="*80
#     print classification_report(y_test_mlb, y_predicted)

    print "="*80
    print "  REPORT FOR FIXED TAG SIZE = %d" % n_suggested_tags
    print "="*80

    precision_micro, recall_micro, f1_micro, _ = metrics.precision_recall_fscore_support(y_test_mlb, y_predicted_fixed_size, average="micro", warn_for=())
    precision_macro, recall_macro, f1_macro, _ = metrics.precision_recall_fscore_support(y_test_mlb, y_predicted_fixed_size, average="macro", warn_for=())

    print "Precision micro: %.3f" % precision_micro
    print "Precision macro: %.3f" % precision_macro
    print "Recall micro: %.3f" % recall_micro
    print "Recall macro: %.3f" % recall_macro
    print "F1 micro: %.3f" % f1_micro
    print "F1 macro: %.3f" % f1_macro

    from evaluation.classification import custom_classification_report
    print custom_classification_report(y_test_mlb, y_predicted_fixed_size, target_names=list(mlb.classes_))

    #unsupervised
    _logger.info("-"*80)
    _logger.info("Unsupervised - Clustering...")
    from unsupervised import clustering
    clustering.cluster(X_train, y_train_mlb, X_test, y_test_mlb, tags, n_suggested_tags, use_numeric_features)

#    _logger.info("-"*80)
#    _logger.info("HAC...")
#    helper.clear_tag_predictions_for_posts(test_posts)
#    hac.hac(len(tags), train_posts, test_posts)
#    evaluation.print_evaluation_results(test_posts)
    return ExitCode.SUCCESS

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


if __name__ == "__main__":
    sys.exit(main())
