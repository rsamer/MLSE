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
from util import helper

_logger = logging.getLogger(__name__)


#################

# libraries
import numpy as np

# scikit-learn base libraries
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

# scikit-learn modules
from sklearn.cross_validation import cross_val_predict
from sklearn.ensemble import VotingClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import log_loss,accuracy_score,mean_squared_error


class StackingClassifier(BaseEstimator,ClassifierMixin):
    '''
    stacking ensemble classifier based on scikit-learn
    '''
    def __init__(self,stage_one_clfs,stage_two_clfs,weights=None, n_runs=10, use_append=True, do_gridsearch=False, params=None, cv=5, scoring="accuracy", print_scores=False):
        '''
        
        weights: weights of the stage_two_clfs
        n_runs: train stage_two_clfs n_runs times and average them (only for probabilistic output)
        '''
        self.stage_one_clfs = stage_one_clfs
        self.stage_two_clfs = stage_two_clfs
        self.n_runs = n_runs
        self.use_append = use_append
        if weights == None:
            self.weights = [1] * len(stage_two_clfs)
        else:
            self.weights = weights
        self.do_gridsearch = do_gridsearch
        self.params = params
        self.cv = cv
        self.scoring = scoring
        self.print_scores = print_scores
    
    def fit(self,X,y):
        '''
        fit the model
        '''
        if self.use_append == True:
            self.__X = X
            self.__y = y
        elif self.use_append == False:
            self.__y = y
            temp = []
            
        # fit the first stage models
        for clf in self.stage_one_clfs:
            y_pred = cross_val_predict(clf[1], X, y, cv=5, n_jobs=1)
            clf[1].fit(X,y)
            y_pred  = np.reshape(y_pred,(len(y_pred),1))
            if self.use_append == True:
                self.__X = np.hstack((self.__X,y_pred))
            elif self.use_append == False:
                temp.append(y_pred)
            
            if self.print_scores == True:
                score = accuracy_score(self.__y,y_pred)
                print("Score of %s: %0.3f" %(clf[0],score))
                
        if self.use_append == False:
            self.__X = np.array(temp).T[0]
            
        # fit the second stage models
        if self.do_gridsearch == False:
            for clf in self.stage_two_clfs:
                clf[1].fit(self.__X,self.__y)      
                
        ### FOR GRIDSEARCH ###  
        else:
            print("GridSearch")
            parameters = {}
            i = 0
            for pair in self.stage_two_clfs:
                est_name = pair[0]
                for key, value in self.params[i].items():
                    key_name = est_name+"__"+key
                    parameters[key_name] = value
                i += 1
                
            majority_voting = VotingClassifier(estimators=self.stage_two_clfs, voting="soft", weights=self.weights)
            grid = GridSearchCV(estimator=majority_voting, param_grid=parameters, cv=self.cv, scoring=self.scoring)
            grid.fit(self.__X, self.__y)
            print()
            print("Best parameters set found on development set:")
            print(grid.best_params_)
            print()
            print("Best score on development set:")
            print(grid.best_score_)
            print()
            print("done")
            
    def predict(self,X_test):
        '''
        predict the class for each sample
        '''
        if self.use_append == True:
            self.__X_test = X_test
        elif self.use_append == False:
            temp = []
        
        # first stage
        for clf in self.stage_one_clfs:
            y_pred = clf[1].predict(X_test)
            y_pred  = np.reshape(y_pred,(len(y_pred),1))
            if self.use_append == True:
                self.__X_test = np.hstack((self.__X_test,y_pred)) 
            elif self.use_append == False:
                temp.append(y_pred)
        
        if self.use_append == False:
            self.__X_test = np.array(temp).T[0]
        
        # second stage
        majority_voting = VotingClassifier(estimators=self.stage_two_clfs, voting="hard", weights=self.weights)
        y_out = majority_voting.predict(self.__X_test)
        return y_out
    
    def predict_proba(self,X_test):
        '''
        predict the probability for each class for each sample
        '''
        if self.use_append == True:
            self.__X_test = X_test
        elif self.use_append == False:
            temp = []
        
        # first stage
        for clf in self.stage_one_clfs:
            y_pred = clf[1].predict(X_test)
            y_pred  = np.reshape(y_pred,(len(y_pred),1))
            if self.use_append == True:
                self.__X_test = np.hstack((self.__X_test,y_pred)) 
            elif self.use_append == False:
                temp.append(y_pred)
            
        if self.use_append == False:
            self.__X_test = np.array(temp).T[0]
        
        # second stage
        preds = []
        for i in range(self.n_runs):
            j = 0
            for clf in self.stage_two_clfs:
                y_pred = clf[1].predict_proba(self.__X_test)  
                preds.append(self.weights[j] * y_pred)
                j += 1
        # average predictions
        y_final = preds.pop(0)
        for pred in preds:
            y_final += pred
        y_out = y_final/(np.array(self.weights).sum() * self.n_runs)
        return y_out      

#################





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
#         'clf__alpha': [0.2, 0.1, 0.06, 0.03, 0.01, 0.001, 0.0001],
}

PARAMS_GRID_SEARCH_TFIDF_FEATURES = {
        'vectorizer__max_features': (2000, 3000, 10000, 20000),
        'vectorizer__max_df': (0.85, ),                            # 1.0)
        'vectorizer__min_df': (2, ),                               #4)
        'vectorizer__ngram_range': ((1, 3), ) #((1, 2), (1, 3))    # unigrams, bigrams or trigrams
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

#     # baseline
#     model = ('clf', OneVsRestClassifier(DummyClassifier("most_frequent"))) # very primitive/simple baseline!
#     model = ('clf', OneVsRestClassifier(MultinomialNB(alpha=.03))) # <-- lidstone smoothing (1.0 would be laplace smoothing!)
# 
#     # "single" classifiers
#     model = ('clf', OneVsRestClassifier(KNeighborsClassifier(n_neighbors=10)))
    model = ('clf', OneVsRestClassifier(SVC(kernel="linear", C=0.025, probability=True)))
#     model = ('clf', OneVsRestClassifier(SVC(kernel="linear", C=2.0, probability=True)))
#     model = ('clf', OneVsRestClassifier(SVC(kernel="rbf", C=0.025, probability=True)))
#     model = ('clf', OneVsRestClassifier(LinearSVC()))
# 
#     # ensemble
#     model = ('clf', OneVsRestClassifier(RandomForestClassifier(n_estimators=200, max_depth=None)))
#     model = ('clf', OneVsRestClassifier(AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), n_estimators=200, learning_rate=1.5, algorithm="SAMME"))

    from sklearn.ensemble import RandomForestClassifier, VotingClassifier
#    clf1 = OneVsRestClassifier(SVC(kernel="linear", C=0.025, probability=True))
    clf1 = OneVsRestClassifier(MultinomialNB(alpha=.01))
    clf2 = OneVsRestClassifier(MultinomialNB(alpha=.03))
    clf3 = OneVsRestClassifier(MultinomialNB(alpha=.01))
#     model = ('clf', OneVsRestClassifier(MultinomialNB(alpha=.03))) # <-- lidstone smoothing (1.0 would be laplace smoothing!)
#    clf2 = OneVsRestClassifier(RandomForestClassifier(n_estimators=10, max_depth=None))
#    clf3 = OneVsRestClassifier(KNeighborsClassifier(n_estimators=7))
#    model = ('clf', VotingClassifier(estimators=[('svc', clf1), ('rf', clf2), ('knn', clf3)], voting='hard')) #voting='soft', weights=[2,1,1])

    clf6 = RandomForestClassifier(n_estimators=1000,max_depth=14,n_jobs=1) # feats = 10
    clf7 = RandomForestClassifier(n_estimators=100,max_depth=14,n_jobs=1) # feats = 10
    #clf7 = GradientBoostingClassifier(n_estimators=100,max_depth=9, max_features=7)  # feats = 7

    first_stage = [('svc', clf1), ('rf', clf2), ('knn', clf3)]
    second_stage = [
                    ("gbm",clf7),
                    ("rf",clf6)
                     ]

    _logger.info(str(model[1]))
    weights = [3,1]
    model = ('clf', OneVsRestClassifier(StackingClassifier(stage_one_clfs=first_stage,stage_two_clfs=second_stage, do_gridsearch=False, weights=weights, n_runs=1, use_append=False, print_scores=True)))
    _classification(model, X_train, y_train, X_test, y_test, mlb, tags, n_suggested_tags,
                    use_numeric_features, do_grid_search)

    #clf1 = RandomForestClassifier(n_estimators=100,random_state=571,max_features=8,max_depth=13,n_jobs=1)
    #clf2 = KNeighborsClassifier(n_neighbors=250, p=1, weights="distance")
    #clf3 = ExtraTreesClassifier(n_estimators=200,max_depth=14, max_features=12,random_state=571,n_jobs=1)
    #clf4 = GaussianNB()
    #clf5 = GradientBoostingClassifier(n_estimators=100,random_state=571,max_depth=6, max_features=7)
    

#############
    
    
#     print("Training")
#     stack.fit(X,y)
#     print("Predict")
#     y_pred = stack.predict_proba(X_test)
#     create_sub(y_pred)
    
#     print("CV")
#     scores = cross_val_score(stack,X,y,scoring="log_loss",cv=skf)
#     print(scores)
#     print("CV-Score: %.3f" % -scores.mean())
    # with append:        Score: 0.783
    # without append:     CV-Score: 0.843
    
#     # gridsearch
#     params1 = {
#                "max_depth": [4],
#                "max_features": [3]
#                }
#     params2 = {
#                "max_depth": [7],
#                "max_features": [4]
#                }
#     paramsset = [params1, params2]
#     stack = StackingClassifier(stage_one_clfs=first_stage,stage_two_clfs=second_stage,weights=weights, n_runs=10, use_append=False,
#                                do_gridsearch=True, params=paramsset, cv=skf, scoring="log_loss", print_scores=False)
#     stack.fit(X,y)






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
