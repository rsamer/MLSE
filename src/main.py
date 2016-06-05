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
import numpy as np
from entities.tag import Tag
from preprocessing import parser, preprocessing as prepr
from sklearn.cross_validation import train_test_split
from supervised import classification
from unsupervised import kmeans, hac
from evaluation import evaluation
from util import helper
from util.docopt import docopt
from util.helper import ExitCode

__version__ = 1.0
_logger = logging.getLogger(__name__)


def usage():
    DEFAULT_TAG_FREQUENCY_THRESHOLD = 3
    DEFAULT_TEST_SIZE = 0.1
    usage = '''
        Automatic Tag Suggestion for StackExchange posts
    
        Usage:
          'main.py' <data-set-path> [--use-caching] [--tag-frequ-thr=<tf>] [--test-size=<ts>]
          'main.py' --version
    
        Options:
          -h --help                     Shows this screen.
          -v --version                  Shows version of this application.
          -c --use-caching              Enables caching in order to avoid redundant preprocessing.
          -f=<tf> --tag-frequ-thr=<tf>  Sets tag frequency threshold -> appropriate value depends on which data set is used! (default=%d).
          -t=<ts> --test-size=<ts>      Sets test size (range: 0.01-0.5) (default=%f).
    ''' % (DEFAULT_TAG_FREQUENCY_THRESHOLD, DEFAULT_TEST_SIZE)
    arguments = docopt(usage)
    kwargs = {}
    kwargs['enable_caching'] = arguments["--use-caching"]
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
    #import os
    logging.basicConfig(
        filename=None, #os.path.join(helper.LOG_PATH, "automatic_tagger.log"),
        level=log_level,
        format='%(asctime)s: %(levelname)7s: [%(name)s]: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def preprocess_tags_and_posts(all_tags, all_posts, tag_frequency_threshold):
    from preprocessing import tags
    filtered_tags = all_tags # f1=0.349
    # FIXME: fails for academia dataset!
    #filtered_tags, all_posts = tags.replace_tag_synonyms(all_tags, all_posts) # f1=0.368, f1=0.352
    filtered_tags = prepr.filter_tags_and_sort_by_frequency(filtered_tags, tag_frequency_threshold)
    prepr.preprocess_tags(filtered_tags)
    posts = prepr.preprocess_posts(all_posts, filtered_tags, filter_posts=True)
    Tag.update_tag_counts_according_to_posts(filtered_tags, posts)
    return filtered_tags, posts


def main():
    kwargs = usage()
    data_set_path = kwargs['data_set_path']
    enable_caching = kwargs['enable_caching']
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
    train_posts, test_posts, _, _ = train_test_split(posts, np.zeros(len(posts)),
                                                     test_size=test_size, random_state=42)
    # TODO: cross-validation!

    # Suggest most frequent tags (baseline)
    # TODO: Random Classifier as baseline => see: http://stats.stackexchange.com/questions/43102/good-f1-score-for-anomaly-detection
#     _logger.info("-"*80)
#     _logger.info("Randomly suggest 2 most frequent tags...")
#     helper.suggest_random_tags(2, test_posts, tags)
#     evaluation.print_evaluation_results(test_posts)
#     _logger.info("-"*80)
#     _logger.info("Only auggest most frequent tag '%s'..." % tags[0])
#     helper.suggest_random_tags(1, test_posts, [tags[0]])
#     evaluation.print_evaluation_results(test_posts)

    # transformation
    _logger.info("-" * 80)
    _logger.info("Transformation...")
    n_features = 20000 #2500  # 2200 # 2500
    from transformation import tfidf, features
    X_train, X_test = tfidf.tfidf(train_posts, test_posts, max_features=n_features, min_df=2)
    #X_train, X_test = features.numeric_features(train_posts, test_posts, tags)

    # 3) learning
    _logger.info("Learning...")

    # supervised
    _logger.info("-"*80)
    _logger.info("Supervised - Classification...")
#    for n_features in range(1000, 3100, 100):

    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import BernoulliNB, MultinomialNB
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.svm import SVC
    from sklearn.svm import LinearSVC
    from sklearn.dummy import DummyClassifier

    classifiers = [
        DummyClassifier("most_frequent"), # very primitive/simple baseline!
        AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), n_estimators=600, learning_rate=1.5, algorithm="SAMME"),
        SVC(kernel="linear", C=0.025, probability=True),
        SVC(kernel="linear", C=0.4, probability=True),
        MultinomialNB(alpha=.03), # <-- lidstone smoothing (1.0 would be laplace smoothing!)

        #KNeighborsClassifier(n_neighbors=10), # f1 = 0.386 (features=2900)
        #RandomForestClassifier(n_estimators=200, max_depth=None, n_jobs=-1),  # f1=0.390
        #SVC(kernel="rbf", C=0.025, probability=True), # penalty = "l2" #"l1"
        #LinearSVC(loss='l2', penalty="l2", dual=False, tol=1e-3),
        #BernoulliNB(alpha=.01)
    ]

    for classifier in classifiers:
        classification.classification(classifier, X_train, X_test, train_posts, test_posts, tags)
        #classification.single_classifier(train_posts, test_posts, tags)
        print "With features: %d" % n_features
        evaluation.print_evaluation_results(test_posts)

#         classification.single_classifier(classifier, X_train, X_test, train_posts, test_posts, tags)
#         #classification.single_classifier(train_posts, test_posts, tags)
#         print "With features: %d" % n_features
#         evaluation.print_evaluation_results(test_posts)

    #unsupervised
#    _logger.info("-"*80)
#    _logger.info("k-Means...")
#    kmeans.kmeans(len(tags), train_posts, test_posts)
#    evaluation.print_evaluation_results(test_posts)
# 
#   _logger.info("-"*80)
#   _logger.info("HAC...")
#   helper.clear_tag_predictions_for_posts(test_posts)
#   hac.hac(len(tags), train_posts, test_posts)
#   evaluation.print_evaluation_results(test_posts)
    return ExitCode.SUCCESS


if __name__ == "__main__":
    sys.exit(main())
