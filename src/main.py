#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
    main -- Main entry point

    main is a function in this module that performs all steps of the entire pipeline
    (preprocessing, training, testing and evaluation)

    usage:      python -m main ../data/example/

    -----------------------------------------------------------------------------
    NOTE: Please have a look at Requirements.txt first.
    NOTE: the data set located in ../data/example/ is NOT a representative subset
          of the entire "programmers.stackexchange.com" data set
    -----------------------------------------------------------------------------

    @author:     Michael Herold, Ralph Samer
    @license:    MIT license
'''

import logging, sys, warnings, numpy as np
from entities.tag import Tag
from preprocessing import tags, parser, preprocessing as prepr
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.cross_validation import train_test_split
from transformation import features
from supervised import classification
from unsupervised import clustering
from util import helper
from util.docopt import docopt
from util.helper import ExitCode

__version__ = 1.0
_logger = logging.getLogger(__name__)

# default values
DEFAULT_TAG_FREQUENCY_THRESHOLD, DEFAULT_N_SUGGESTED_TAGS, DEFAULT_TEST_SIZE = (3, 2, 0.1)


def preprocess_tags_and_posts(all_tags, all_posts, tag_frequency_threshold, enable_stemming=True,
                              replace_adjacent_tag_occurences=True,
                              replace_token_synonyms_and_remove_adjacent_stopwords=True):
    # FIXME: figure out why this fails on academia dataset!
    #filtered_tags = all_tags # f1=0.349
    filtered_tags, all_posts = tags.replace_tag_synonyms(all_tags, all_posts)
    filtered_tags = prepr.filter_tags_and_sort_by_frequency(filtered_tags, tag_frequency_threshold)
    if enable_stemming is True:
        prepr.stem_tags(filtered_tags)
    posts = prepr.preprocess_posts(all_posts, filtered_tags, True, enable_stemming,
                                   replace_adjacent_tag_occurences,
                                   replace_token_synonyms_and_remove_adjacent_stopwords)
    Tag.update_tag_counts_according_to_posts(filtered_tags, posts)
    return filtered_tags, posts


def main(data_set_path, enable_caching, use_numeric_features, n_suggested_tags,
         tag_frequency_threshold, test_size, test_with_training_data):

    helper.setup_logging(logging.INFO)
    helper.make_dir_if_not_exists(helper.CACHE_PATH)
    _logger.info("Caching ON!" if enable_caching else "Caching OFF!")

    #===============================================================================================
    # 1) Parsing
    #-----------------------------------------------------------------------------------------------
    _logger.info("Parsing...")
    all_tags, all_posts, cache_file_name_prefix = parser.parse_tags_and_posts(data_set_path)
    all_posts_assignments = reduce(lambda x,y: x + y, map(lambda p: len(p.tag_set), all_posts))
    tag_frequency_threshold = tag_frequency_threshold
    cache_file_name_prefix += "_%d_" % tag_frequency_threshold # caching depends on tag-frequency

    #===============================================================================================
    # 2) Preprocessing
    #-----------------------------------------------------------------------------------------------
    _logger.info("Preprocessing...")
    if not enable_caching or not helper.cache_exists_for_preprocessed_tags_and_posts(cache_file_name_prefix):
        tags, posts = preprocess_tags_and_posts(all_tags, all_posts, tag_frequency_threshold,
                                                enable_stemming=False, replace_adjacent_tag_occurences=True,
                                                replace_token_synonyms_and_remove_adjacent_stopwords=True)
        helper.write_preprocessed_tags_and_posts_to_cache(cache_file_name_prefix, tags, posts)
    else:
        _logger.info("CACHE HIT!")
        tags, posts = helper.load_preprocessed_tags_and_posts_from_cache(cache_file_name_prefix)

    helper.print_tags_and_posts_summary(len(all_tags), len(tags), all_posts, all_posts_assignments, posts)

    #===============================================================================================
    # 3) Split training and test data
    #-----------------------------------------------------------------------------------------------
    _logger.info("Splitting data set -> Training: {}%, Test: {}%".format((1-test_size)*100, test_size*100))

    y = map(lambda p: tuple(map(lambda t: t.name, p.tag_set)), posts)
    if not use_numeric_features:
        X = map(lambda p: ' '.join(p.tokens(title_weight=3)), posts)
        assert len(X) == len(y)
    else:
        X, _ = features.numeric_features(posts, [], tags, True)
        assert X.shape[0] == len(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    from transformation import tfidf#, features
    X_train, X_test = tfidf.tfidf(X_train, X_test, max_features=400, min_df=4)
    #X_train, X_test = features.numeric_features(train_posts, test_posts, tags)

    #===============================================================================================
    # 4) prepare, vectorize and transform
    #-----------------------------------------------------------------------------------------------
    _logger.info("-" * 80)
    _logger.info("Vectorization and Transformation...")
    mlb = MultiLabelBinarizer()
    y_train_mlb = mlb.fit_transform(np.array(y_train))
    y_test_mlb = mlb.transform(np.array(y_test))
    warnings.filterwarnings('ignore')

    # check if should test with training data (default: False)
    if test_with_training_data is True:
        _logger.warn("-"*80)
        _logger.warn("   !!! Testing with training data !!!")
        _logger.warn("-"*80)
        X_test, y_test, y_test_mlb = X_train, y_train, y_train_mlb

    #===============================================================================================
    # 5) Supervised - Classification
    #-----------------------------------------------------------------------------------------------
    _logger.info("-"*80)
    _logger.info("Supervised - Classification...")
    classification.classification(X_train, y_train_mlb, X_test, y_test_mlb, mlb, tags,
                                  n_suggested_tags, use_numeric_features)

    ####
    ####
    ####
    sys.exit()
    ####
    ####
    ####

    #===============================================================================================
    # 6) Unsupervised - Clustering
    #-----------------------------------------------------------------------------------------------
    _logger.info("-"*80)
    _logger.info("Unsupervised - Clustering...")
    clustering.clustering(X_train, y_train_mlb, X_test, y_test_mlb, tags, n_suggested_tags,
                          use_numeric_features)
    return ExitCode.SUCCESS


def usage():
    usage = '''
        Automatic Tag Suggestion for StackExchange posts
    
        Usage:
          'main.py' <data-set-path> [--use-caching] [--use-numeric-features] ''' + \
                 '''[--test-with-training-data] [--num-suggested-tags=<nt>] ''' + \
                 '''[--tag-frequ-thr=<tf>] [--test-size=<ts>]
          'main.py' --version

        Options:
          -h --help                         Shows this screen
          -v --version                      Shows version of this application
          -c --use-caching                  Enables caching in order to avoid redundant preprocessing
          -n --use-numeric-features         Enables numeric features (PMI) instead of TFxIDF
          -d --test-with-training-data      Test with training data instead of test data
          -s=<nt> --num-suggested-tags=<nt> Number of suggested tags (default=%d)
          -f=<tf> --tag-frequ-thr=<tf>      Sets tag frequency threshold -> appropriate value depends on which data set is used! (default=%d)
          -t=<ts> --test-size=<ts>          Sets test size (range: 0.01-0.5) (default=%f)
    ''' % (DEFAULT_N_SUGGESTED_TAGS, DEFAULT_TAG_FREQUENCY_THRESHOLD, DEFAULT_TEST_SIZE)
    arguments = docopt(usage)
    kwargs = {}
    kwargs['enable_caching'] = bool(arguments["--use-caching"])
    kwargs['use_numeric_features'] = bool(arguments["--use-numeric-features"])
    kwargs['test_with_training_data'] = bool(arguments["--test-with-training-data"])
    kwargs['n_suggested_tags'] = int(arguments["--num-suggested-tags"][0]) if arguments["--num-suggested-tags"] else DEFAULT_N_SUGGESTED_TAGS
    kwargs['tag_frequency_threshold'] = int(arguments["--tag-frequ-thr"][0]) if arguments["--tag-frequ-thr"] else DEFAULT_TAG_FREQUENCY_THRESHOLD
    kwargs['test_size'] = max(0.01, min(0.5, float(arguments["--test-size"][0]))) if arguments["--test-size"] else DEFAULT_TEST_SIZE
    kwargs['data_set_path'] = arguments["<data-set-path>"]
    if arguments["--version"]:
        print("Automatic Tag Suggestion for StackExchange posts\nVersion: {}".format(__version__))
        sys.exit(ExitCode.SUCCESS)
    return kwargs


if __name__ == "__main__":
    kwargs = usage()
    sys.exit(main(**kwargs))
