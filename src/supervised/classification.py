# -*- coding: utf-8 -*-

import logging
from time import time
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.dummy import DummyClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression
from evaluation.classification import custom_classification_report
import ensemble
from util import helper

_logger = logging.getLogger(__name__)


#===================================================================================================
# CONFIGURATION
#===================================================================================================

def extract_tokens(text): return text.split()

PARAMS_COUNT_VECTORIZER_COMMON = {
        'input': 'content',
        'tokenizer': extract_tokens,
        'preprocessor': None,
        'analyzer': 'word',
        'encoding': 'utf-8',
        'decode_error': 'strict',
        'strip_accents': None,
        'lowercase': True,
        'stop_words': None,
        'vocabulary': None,
        'binary': False
}


PARAMS_COUNT_VECTORIZER_NO_GRID_SEARCH = {
        #------------------------------------------------------------------------------------------
        # NOTE: parameters of best estimator determined by grid search go here:
        #------------------------------------------------------------------------------------------
        'max_features': 2000,
        'max_df': 0.85,
        'min_df': 2,
        'ngram_range': (1, 3),
}


DEFAULT_PARAMS_TFIDF_TRANSFORMER = {
        'norm': 'l2',
        'sublinear_tf': False,
        'smooth_idf': True,
        'use_idf': True
}

PARAMS_GRID_SEARCH_COMMON = {
#       # insert classifier parameters here!
#         'clf__alpha': [0.2, 0.1, 0.06, 0.03, 0.01, 0.001, 0.0001],
}

PARAMS_GRID_SEARCH_TFIDF_FEATURES = {
        'vectorizer__max_features': (None, 2000, 3000, 10000, 20000),
        'vectorizer__max_df': (0.7, 0.85, 1.0),
        'vectorizer__min_df': (2, 3, 4),
        'vectorizer__ngram_range': ((1,1), (1,2), (1,3)) # unigrams, bigrams or trigrams mixed
}
#===================================================================================================


def _classification(model, X_train, y_train, X_test, y_test, mlb, tags, n_suggested_tags,
                    use_numeric_features, do_grid_search=False):
    #-----------------------------------------------------------------------------------------------
    # SETUP PARAMETERS & CLASSIFIERS
    #-----------------------------------------------------------------------------------------------
    grid_search_params = PARAMS_GRID_SEARCH_COMMON
    if not use_numeric_features:
        parameters_count_vectorizer = PARAMS_COUNT_VECTORIZER_COMMON
        if do_grid_search:
            grid_search_params = helper.merge_two_dicts(PARAMS_GRID_SEARCH_COMMON, PARAMS_GRID_SEARCH_TFIDF_FEATURES)
        else:
            parameters_count_vectorizer = helper.merge_two_dicts(parameters_count_vectorizer,
                                                                 PARAMS_COUNT_VECTORIZER_NO_GRID_SEARCH)

        classifier = Pipeline([
            ('vectorizer', CountVectorizer(**parameters_count_vectorizer)),
            ('tfidf', TfidfTransformer(**DEFAULT_PARAMS_TFIDF_TRANSFORMER)),
            model
        ])
    else:
        classifier = Pipeline([model])

    #-----------------------------------------------------------------------------------------------
    # LEARNING / FIT CLASSIFIER / GRIDSEARCH VIA CROSS VALIDATION (OF TRAINING DATA)
    #-----------------------------------------------------------------------------------------------
    if do_grid_search:
        _logger.info("-"*80)
        _logger.info("Grid search")
        _logger.info("-"*80)
        classifier = GridSearchCV(classifier, grid_search_params, n_jobs=-1, cv=3, verbose=1, scoring='f1_micro')
        _logger.info("Parameters: %s", grid_search_params)
    elif not use_numeric_features:
        _logger.info("Count vectorizer parameters: %s", PARAMS_COUNT_VECTORIZER_NO_GRID_SEARCH)

    t0 = time()
    classifier.fit(np.array(X_train) if not use_numeric_features else X_train, y_train)
    _logger.info("Done in %0.3fs" % (time() - t0))

    if do_grid_search:
        _logger.info("Best parameters set:")
        best_parameters = classifier.best_estimator_.get_params()
        for param_name in sorted(grid_search_params.keys()):
            _logger.info("\t%s: %r" % (param_name, best_parameters[param_name]))

    #-----------------------------------------------------------------------------------------------
    # PREDICTION
    #-----------------------------------------------------------------------------------------------
    _logger.info("Number of suggested tags: %d" % n_suggested_tags)
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
#     # NOTE: uncomment these lines if you want variable tag size
#     y_predicted = classifier.predict(np.array(X_test) if not use_numeric_features else X_test)
#     print "-"*80
#     print y_predicted
#     for item, labels in zip(X_test, mlb.inverse_transform(y_predicted)):
#         print '%s -> (%s)' % (item[:40], ', '.join(labels))
#    
#     print "="*80
#     print "  REPORT FOR VARIABLE TAG SIZE"
#     print "="*80
#     print custom_classification_report(y_test, y_predicted, target_names=list(mlb.classes_))
#  
#     if not use_numeric_features:
#         print "-"*80
#         for item, labels in zip(X_test, y_predicted_label_list):
#             print '%s -> (%s)' % (item[:40], ', '.join(labels))

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


def classification(X_train, y_train, X_test, y_test, mlb, tags, n_suggested_tags,
                   use_numeric_features, do_grid_search=False):

    #-----------------------------------------------------------------------------------------------
    # NOTE: in order to use a classifier:
    #
    #       1) uncomment which classifier to choose
    #       2) when NOT using numeric features and NOT doing grid search:
    #          Please also modify PARAMS_COUNT_VECTORIZER_NO_GRID_SEARCH according to your needs.
    #
    #-----------------------------------------------------------------------------------------------

    models = [
        #-------------------------------------------------------------------------------------------
        # baseline
        #-------------------------------------------------------------------------------------------
#       ('clf', OneVsRestClassifier(DummyClassifier("most_frequent"))), # very primitive baseline!
        ('clf', OneVsRestClassifier(MultinomialNB(alpha=.1))), # <-- lidstone smoothing (1.0 == laplace smoothing!)
        #-------------------------------------------------------------------------------------------

        #-------------------------------------------------------------------------------------------
        # "single" classifiers
        #-------------------------------------------------------------------------------------------
        # larger penalty parameter works better for numeric features
        ('clf', OneVsRestClassifier(SVC(kernel="linear", C=2.0, tol=0.001, probability=True))),
        # smaller tolerance (1e-4) works slightly better for numeric features
        ('clf', OneVsRestClassifier(SVC(kernel="linear", C=2.0, tol=0.0001, probability=True))),
        # smaller penalty parameter works better for TFIDF
        ('clf', OneVsRestClassifier(SVC(kernel="linear", C=0.025, probability=True))),
        ('clf', OneVsRestClassifier(KNeighborsClassifier(n_neighbors=10))),
        ('clf', OneVsRestClassifier(SVC(kernel="rbf", C=0.025, probability=True))),
#         ('clf', OneVsRestClassifier(LinearSVC())),
        #-------------------------------------------------------------------------------------------

        #-------------------------------------------------------------------------------------------
        # ensemble
        #-------------------------------------------------------------------------------------------
        ('clf', ensemble.LabelFrequencyBasedVotingClassifier(
            ('clf1', OneVsRestClassifier(KNeighborsClassifier(n_neighbors=10))),
            ('clf2', OneVsRestClassifier(SVC(kernel="linear", C=0.025, probability=True))),
            weights=None
        )),
        ('clf', ensemble.CustomVotingClassifier(
            [
                OneVsRestClassifier(SVC(kernel="linear", C=2.0, probability=True)),
                OneVsRestClassifier(KNeighborsClassifier(n_neighbors=10)),
                OneVsRestClassifier(SVC(kernel="rbf", C=0.025, probability=True))
            ],
            weights = [.5, .2, .3]
        )),
        ('clf', OneVsRestClassifier(BaggingClassifier(base_estimator=DecisionTreeClassifier(max_depth=10), n_estimators=40, random_state=None))),
        ('clf', OneVsRestClassifier(RandomForestClassifier(n_estimators=200, max_depth=None))),
        ('clf', OneVsRestClassifier(AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), n_estimators=200, learning_rate=1.5, algorithm="SAMME")))
        #-------------------------------------------------------------------------------------------
    ]
    if not do_grid_search:
        # this is because StackedGeneralizer uses its own CrossValidation
        models += [('clf', ensemble.StackedGeneralizer(
            # base models
            [
                OneVsRestClassifier(SVC(kernel="linear", C=2.0, probability=True)),
                OneVsRestClassifier(KNeighborsClassifier(n_neighbors=10)),
                OneVsRestClassifier(SVC(kernel="rbf", C=0.025, probability=True))
            ],
            # blending model
            OneVsRestClassifier(LogisticRegression()),
            n_folds = 3
        ))]


    for model in models:
        _classification(model, X_train, y_train, X_test, y_test, mlb, tags, n_suggested_tags,
                        use_numeric_features, do_grid_search)

