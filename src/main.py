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
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.base import TransformerMixin
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.multiclass import OneVsRestClassifier
from sklearn.cross_validation import StratifiedKFold
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
    import platform
    #import os
    logging.basicConfig(
        filename=None, #os.path.join(helper.LOG_PATH, "automatic_tagger.log"),
        level=log_level,
        format='%(asctime)s: %(levelname)7s: [%(name)s]: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    # based on: http://stackoverflow.com/a/1336640
    if platform.system() == 'Windows':
        # Windows does not support ANSI escapes and we are using API calls to set the console color
        logging.StreamHandler.emit = helper.add_coloring_to_emit_windows(logging.StreamHandler.emit)
    else:
        # all non-Windows platforms are supporting ANSI escapes so we use them
        logging.StreamHandler.emit = helper.add_coloring_to_emit_ansi(logging.StreamHandler.emit)
        #log = logging.getLogger()
        #log.addFilter(log_filter())
        #//hdlr = logging.StreamHandler()
        #//hdlr.setFormatter(formatter())


def preprocess_tags_and_posts(all_tags, all_posts, tag_frequency_threshold):
    from preprocessing import tags
    # FIXME: figure out why this fails on academia dataset!
    #filtered_tags = all_tags # f1=0.349
    filtered_tags, all_posts = tags.replace_tag_synonyms(all_tags, all_posts) # f1=0.368, f1=0.352
    filtered_tags = prepr.filter_tags_and_sort_by_frequency(filtered_tags, tag_frequency_threshold)
    prepr.preprocess_tags(filtered_tags)
    posts = prepr.preprocess_posts(all_posts, filtered_tags, filter_posts=True)
    Tag.update_tag_counts_according_to_posts(filtered_tags, posts)
    return filtered_tags, posts

# TODO: check if this is still needed...
def extract_tokens(tokens):
    return tokens

# class MultilabelClassifierWrapper(TransformerMixin):
#     def __init__(self, clf):
#         self.clf = clf()
# 
#     def transform(self, X, **transform_params):
#         return self.clf.transform(X, transform_params)
# 
#     def fit(self, X, y=None, **fit_params):
#         print len(X)
#         sys.exit()
#         return self.clf.fit(X, y, fit_params)
# 
#     def score(self, X, y, sample_weight=None):
#         return self.clf.score(X, y, sample_weight)
# 
#     def get_params(self, deep=True):
#         return self.clf.get_params(deep)
# 
#     def set_params(self, **params):
#         return self.clf.set_params(params)

class MyNaiveBayes(MultinomialNB):
    def __init__(self, alpha=1.0, fit_prior=True, class_prior=None):
        super(MyNaiveBayes, self).__init__(alpha, fit_prior, class_prior)
    def fit(self, X, y, sample_weight=None):
        print X
        print "HELLO!"
        sys.exit()


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

    X = map(lambda p: ' '.join(p.tokens()), posts)
    y = map(lambda p: tuple(map(lambda t: t.name, p.tag_set)), posts)
    assert len(X) == len(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    mlb = MultiLabelBinarizer()
    y_train_mlb = mlb.fit_transform(np.array(y_train))
    y_test_mlb = mlb.transform(np.array(y_test))

    # transformation
#     _logger.info("-" * 80)
#     _logger.info("Transformation...")
#     n_features = 20000 #2500  # 2200 # 2500 for KNN
#     from transformation import tfidf, features
#     X_train, X_test = tfidf.tfidf(train_posts, test_posts, max_features=None, min_df=2)
    #X_train, X_test = features.numeric_features(train_posts, test_posts, tags)

    # 3) learning
    _logger.info("Learning...")

    # supervised
    _logger.info("-"*80)
    _logger.info("Supervised - Classification...")
    t0 = time()

    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
    pipeline = Pipeline([
        ('vectorizer', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', OneVsRestClassifier(LinearSVC()))])

#     pipeline = make_pipeline(
#         TfidfVectorizer(
#             stop_words=None,
#             preprocessor=extract_tokens,
#             analyzer=extract_tokens,
#             tokenizer=extract_tokens,
#             #token_pattern=r'.*',
#             smooth_idf=False,
#             sublinear_tf=False,
#             norm=None
#         ),
# #         MultilabelClassifierWrapper(MultinomialNB(alpha=0.1))
# #        OneVsRestClassifier(MyNaiveBayes(alpha=0.1))
#         OneVsRestClassifier(SVC(kernel="linear", C=0.025))
#     )
#     parameters = {
#         'tfidfvectorizer__max_df': (0.75, 0.8, 1.0),
#         'tfidfvectorizer__ngram_range': ((1, 1), (1, 2), (1, 3)), # unigrams, bigrams or trigrams
#         'tfidfvectorizer__use_idf': (True, False),
#         #'tfidfvectorizer__norm': ('l1', 'l2'),
#         'tfidfvectorizer__min_df': (1, 2, 3, 4),
#         'tfidfvectorizer__max_features': (None, 1000, 2000, 3000, 5000, 10000, 15000),
#         #'estimator__alpha': [0.2, 0.1, 0.06, 0.03, 0.01, 0.001, 0.0001],
#     }
    parameters = {
        'vectorizer__max_features': (None, 1000, 2000, 3000, 5000, 10000, 15000),
        'vectorizer__max_df': (0.75, 0.8, 1.0),
        'vectorizer__min_df': (1, 2, 3, 4),
        'vectorizer__ngram_range': ((1, 1), (1, 2), (1, 3)), # unigrams, bigrams or trigrams
        'tfidf__use_idf': (True, False),
        #'tfidf__norm': ('l1', 'l2'),
        #'estimator__alpha': [0.2, 0.1, 0.06, 0.03, 0.01, 0.001, 0.0001],
    }
#     #scores = ['precision', 'recall']
#     #for score in scores:
#     #_logger.info("# Tuning hyper-parameters for %s", score)
#     clf = GridSearchCV(pipeline, parameters, n_jobs=1,#-1,
#                                cv=3,#StratifiedKFold(y=y_train_mlb, n_folds=3),
#                                verbose=1)#, scoring='%s_weighted' % score)
#     _logger.info("Parameters: %s", parameters)
#     clf.fit(X_train, y_train_mlb)
#     _logger.info("Done in %0.3fs" % (time() - t0))
# 
#     _logger.info("Best score: %0.3f", clf.best_score_)
#     _logger.info("Best parameters set:")
#     best_parameters = clf.best_estimator_.get_params()
#     for param_name in sorted(parameters.keys()):
#         _logger.info("\t%s: %r" % (param_name, best_parameters[param_name]))
# 
#     y_pred_mlb = clf.predict(X_test)
#     print classification_report(mlb.transform(y_test), y_pred_mlb)
#     print mlb.inverse_transform(mlb.transform(y_pred_mlb))
#     #print classification_report(y_test, y_pred_mlb)
#     sys.exit()

    classifiers = [
#         DummyClassifier("most_frequent"), # very primitive/simple baseline!
#        AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), n_estimators=200, learning_rate=1.5, algorithm="SAMME"),
#        SVC(kernel="linear", C=0.4, probability=True),
        #KNeighborsClassifier(n_neighbors=10), # f1 = 0.386 (features=2900)
        #RandomForestClassifier(n_estimators=200, max_depth=None, n_jobs=-1),  # f1=0.390
        MultinomialNB(alpha=.03), # <-- lidstone smoothing (1.0 would be laplace smoothing!)
        #BernoulliNB(alpha=.01)
        SVC(kernel="linear", C=0.025, probability=True),
        SVC(kernel="rbf", C=0.025, probability=True), # penalty = "l2" #"l1"
        #LinearSVC(loss='l2', penalty="l2", dual=False, tol=1e-3),
    ]

    # TODO: remove single_classifier in classification.py -> http://stackoverflow.com/a/31586026
    classifier = Pipeline([
        ('vectorizer', CountVectorizer()),#min_df=2, max_features=None, ngram_range=(1, 3))),
        ('tfidf', TfidfTransformer()),
        #('clf', OneVsRestClassifier(SVC(kernel="linear", probability=True)))])
        ('clf', OneVsRestClassifier(SVC(kernel="linear", C=0.025, probability=True)))])
#        ('clf', OneVsRestClassifier(MultinomialNB(alpha=.03)))])


    n_suggested_tags = 2 # FIXME: use this as GridSearchCV parameter!!

    # [  TRAINING    | TEST  ]
    # [ 1  |  2 |  3 | TEST  ]
    # [ 1  |  2 |  x | TEST  ]
    # [ 1  |  x |  3 | TEST  ]
    # [ x  |  2 |  3 | TEST  ]
    #classifier = GridSearchCV(classifier, parameters, n_jobs=-1, cv=3, verbose=0)


    classifier.fit(np.array(X_train), y_train_mlb)


#     _logger.info("Best parameters set:")
#     best_parameters = classifier.best_estimator_.get_params()
#     for param_name in sorted(parameters.keys()):
#         _logger.info("\t%s: %r" % (param_name, best_parameters[param_name]))


    y_predicted = classifier.predict(np.array(X_test))
    y_predicted_probab = classifier.predict_proba(np.array(X_test))
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
    from sklearn import metrics
    print "Precision micro: %.3f" % metrics.precision_score(y_test_mlb, y_predicted_fixed_size, average="micro")
    print "Precision macro: %.3f" % metrics.precision_score(y_test_mlb, y_predicted_fixed_size, average="macro")
    print "Recall micro: %.3f" % metrics.recall_score(y_test_mlb, y_predicted_fixed_size, average="micro")
    print "Recall macro: %.3f" % metrics.recall_score(y_test_mlb, y_predicted_fixed_size, average="macro")
    print "F1 micro: %.3f" % metrics.f1_score(y_test_mlb, y_predicted_fixed_size, average="micro")
    print "F1 macro: %.3f" % metrics.f1_score(y_test_mlb, y_predicted_fixed_size, average="macro")
    print classification_report(y_test_mlb, y_predicted_fixed_size, target_names=list(mlb.classes_))


#     for clf in classifiers:
#         print y_train_mlb
#         one_vs_rest_clf, _ = classification.one_vs_rest(clf, np.array(X_train), y_train_mlb)
#         y_pred_mlb = one_vs_rest_clf.predict(np.array(X_test))
#         print classification_report(mlb.transform(y_test), y_pred_mlb)

#         classification.classification(classifier, X_train, X_test, train_posts, test_posts, tags)
#         #classification.single_classifier(train_posts, test_posts, tags)
#         evaluation.print_evaluation_results(test_posts)

#         # sanity checks!
#         assert classifier.classes_[0] == False
#         assert classifier.classes_[1] == True
#         for p1, p2 in prediction_probabilities_list:
#             assert abs(1.0 - (p1+p2)) < 0.001

#         classification.single_classifier(classifier, X_train, X_test, train_posts, test_posts, tags)
        #classification.single_classifier(train_posts, test_posts, tags)
#         print "With features: %d" % n_features
#         evaluation.print_evaluation_results(test_posts)

    #unsupervised
#    _logger.info("-"*80)
#    _logger.info("k-Means...")
#    kmeans.kmeans(len(tags), train_posts, test_posts)
#    evaluation.print_evaluation_results(test_posts)
# 
#    _logger.info("-"*80)
#    _logger.info("HAC...")
#    helper.clear_tag_predictions_for_posts(test_posts)
#    hac.hac(len(tags), train_posts, test_posts)
#    evaluation.print_evaluation_results(test_posts)
    return ExitCode.SUCCESS

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


if __name__ == "__main__":
    sys.exit(main())
