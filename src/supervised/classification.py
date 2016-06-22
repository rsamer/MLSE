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

#    less_frequent_tags = [".net", "algorithms", "architecture", "asp.net", "books", "bug", "business", "c#", "c++", "cloud-computing", "code-quality", "code-reviews", "coding", "communication", "comparison", "database", "debugging", "developer-tools", "ethics", "exceptions", "experience", "frameworks", "freelancing", "functional-programming", "gpl", "hiring", "interview", "issue-tracking", "java", "job-market", "language-design", "legal", "licensing", "maintenance", "management", "math", "motivation", "non-programmers", "object-oriented", "orm", "paradigms", "php", "pricing", "programming-practices", "project", "python", "refactoring", "resume", "scala", "skills", "software-developer", "source-code", "sql", "syntax", "tdd", "team", "terminology", "testing", "uml", "users", "variables", "visual-studio", "web-applications", "websites"]
    less_frequent_tags = ["3d", "abstraction", "acceptance-testing", "access", "access-modifiers", "accessibility", "actionscript", "actor-model", "ajax", "amazon-ec2", "analysis", "anti-patterns", "apache", "apache-license", "app", "apple", "applications", "appstore", "architect", "array", "artificial-intelligence", "aspect-oriented", "assembly", "assertions", "async", "authentication", "automation", "backups", "backward-compatibility", "bad-code", "basic", "bdd", "behavior", "benchmarking", "beta", "big-o", "billing", "binary", "bitwise-operators", "browser", "browser-compatibility", "bug-report", "business", "business-logic", "business-process", "buzzwords", "caching", "cakephp", "captcha", "change", "changes", "character-encoding", "class", "clean-code", "client-side", "clojure", "closed-source", "closures", "cloud-computing", "cluster", "cms", "co-workers", "cobol", "code-analysis", "code-generation", "code-organization", "code-ownership", "code-reuse", "code-smell", "codeigniter", "coding", "collaboration", "command-line", "commercial", "community", "company", "comparison", "complexity", "composition", "concepts", "conditions", "conferences", "configuration", "configuration-management", "continuous-integration", "contract", "contribution", "control-structures", "copy-paste-programming", "copyright", "cost-estimation", "coupling", "cowboy-coding", "cpu", "cqrs", "creativity", "crm", "cross-platform", "culture", "customer-relations", "data", "data-mining", "data-structures", "data-types", "deadlines", "debugging", "decisions", "definition", "delphi", "dependency-injection", "dependency-management", "desktop-development", "developer-tools", "diagrams", "difference", "distributed-computing", "distribution", "django", "dll", "document", "domain-model", "drupal", "dsl", "dvcs", "e-commerce", "ebook", "eclipse", "efficiency", "emacs", "email", "embedded-systems", "employment", "encapsulation", "encryption", "end-users", "engineering", "enterprise-architecture", "enterprise-development", "entity-framework", "entrepreneurship", "erlang", "erp", "etiquette", "etymology", "evaluation", "event-programming", "extensibility", "extreme-programming", "f#", "facebook", "failure", "features", "feedback", "file-handling", "file-storage", "file-structure", "file-systems", "final", "financial", "finite-state-machine", "firefox", "flash", "flex", "flowchart", "forms", "fortran", "free-time", "freeware", "front-end", "functions", "garbage-collection", "generics", "go", "google", "google-app-engine", "gpu", "grails", "grammar", "graphics", "groovy", "gwt", "hacking", "hardware", "hashing", "headers", "heap", "hibernate", "hiring", "hosting", "ideas", "idioms", "iis", "image-processing", "immutability", "imperative-programming", "implementations", "indentation", "indexing", "industry", "information", "inheritance", "installer", "integration-testing", "intellectual-property", "interfaces", "internationalization", "internet", "internet-explorer", "internship", "interpreters", "ioc-containers", "ipad", "iterative-development", "iterator", "jenkins", "jit", "job-market", "joel-test", "jre", "jruby", "json", "jsp", "kanban", "kernel", "knowledge", "knowledge-base", "knowledge-management", "knowledge-transfer", "lambda", "lamp", "language-agnostic", "language-choice", "language-discussion", "language-features", "languages", "law", "lean", "legacy", "libraries", "linq", "localization", "logging", "logic-programming", "low-level", "mac", "machine-learning", "maintainability", "maintenance", "managers", "marketing", "math", "matlab", "measurement", "meetings", "memory", "mentor", "messaging", "meta-programming", "methodology", "methods", "metrics", "microsoft", "migration", "mistakes", "mit-license", "mobile", "mocking", "modeling", "modules", "mono", "motivation", "mvp", "mvvm", "n-tier", "namespace", "networking", "nhibernate", "node.js", "non-disclosure-agreement", "non-programmers", "non-technical-manager", "nosql", "notation", "null", "numbers", "obfuscation", "object-oriented-design", "objective-c", "opengl", "openid", "operating-systems", "operators", "oracle", "organization", "osx", "packages", "paradigms", "pascal", "patents", "patterns-and-practices", "paying", "perl", "personal-projects", "philosophy", "planning", "platforms", "plugins", "pointers", "politics", "portability", "portfolio", "porting", "pragmatism", "presentation", "pricing", "privacy", "problem-solving", "product", "product-features", "product-management", "product-owner", "production", "profession", "professional-development", "profiling", "programming-logic", "programming-practices", "project", "project-hosting", "project-planning", "project-structure", "projects-and-solutions", "prolog", "pronunciation", "proprietary", "protocol", "prototyping", "pseudocode", "puzzles", "qa", "qt", "quality", "quotations", "random", "rational-unified-process", "recruiting", "reference", "regular-expressions", "release", "release-management", "reporting", "repository", "research", "resources", "rest", "reverse-engineering", "rewrite", "robotics", "runtime", "sales", "scala", "scalability", "scheduling", "schema", "scheme", "scope", "scripting", "sdlc", "search", "selenium", "semantics", "sequence", "server", "server-side", "services", "single-responsibility", "singleton", "soa", "soap", "social", "social-networks", "social-skills", "sockets", "software-as-a-service", "software-craftsmanship", "software-distribution", "software-patent", "software-updates", "solid", "solo-development", "sorting", "source-code", "spring", "sprint", "standardization", "standards", "static-access", "static-analysis", "static-methods", "statistics", "storage", "stored-procedures", "stories", "strategy", "support", "swing", "switch-statement", "synchronization", "system", "system-reliability", "systems", "systems-programming", "teaching", "team", "team-building", "team-foundation-server", "team-leader", "technical-debt", "technical-support", "technique", "technology", "telecommuting", "templates", "terms-of-service", "text-editor", "theory", "third-party-libraries", "time", "time-management", "time-tracking", "tomcat", "tracking", "training", "transaction", "tsql", "twitter", "unix", "upgrade", "usability", "user-experience", "user-story", "users", "validation", "variables", "vb.net", "verification", "vim", "virtual-machine", "virtualization", "visual-basic", "vulnerabilities", "waterfall", "wcf", "web", "web-applications", "web-browser", "web-crawler", "web-framework", "web-hosting", "webforms", "websites", "wiki", "windows-phone-8", "winforms", "workflows", "wpf", "writing", "xaml", "xcode", "zend-framework"]
    import copy
    classes = copy.copy(mlb.classes_)
    assert classes is not None

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
                OneVsRestClassifier(SVC(kernel="linear", C=2.0, tol=0.001, probability=True)),
                OneVsRestClassifier(KNeighborsClassifier(n_neighbors=10))
            ],
            weights = [.5, .5]
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
        print model[1]
        _classification(model, X_train, y_train, X_test, y_test, mlb, tags, n_suggested_tags,
                        use_numeric_features, do_grid_search)

