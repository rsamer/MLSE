#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
main -- Main entry point

main is a module that invokes the training process for
given StackExchange data set

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
from sklearn.cross_validation import train_test_split

from entities.tag import Tag
from evaluation import evaluation
from preprocessing import parser
from preprocessing import preprocessing as prepr
from util import helper
from util.helper import ExitCode
from supervised import naive_bayes
from unsupervised import kmeans
from util.docopt import docopt

__version__ = 1.0
_logger = logging.getLogger(__name__)


def usage():
    usage = '''Automatic Tag Suggestion for StackExchange posts

    Usage:
      'main.py' <data-set-path> [--use-caching] [--tag-frequ-thr=<tf>]
      'main.py' --version

    Options:
      -h --help                     Shows this screen.
      -v --version                  Shows version of this application.
      -c --use-caching              Enables caching in order to avoid redundant preprocessing.
      -t=<tf> --tag-frequ-thr=<tf>  Sets tag frequency threshold -> appropriate value depends on which data set is used! (default=3).
    '''
    arguments = docopt(usage)
    kwargs = {}
    kwargs['enable_caching'] = arguments["--use-caching"]
    kwargs['tag_frequency_threshold'] = int(arguments["--tag-frequ-thr"]) if arguments["--tag-frequ-thr"] else 3
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


# TODO: move this to tests!!
# DEBUG BEGIN
#test_post1 = Post(1, "", u"RT @marcobonzanini: just, an example! :D http://example.com/what?q=test #NLP", set(), 100)
#test_post2 = Post(2, "", u"0x2AF3 #143152 A b C d e f g h i j k f# u# and C++ is a test hehe wt iop complicated programming-languages object oriented object-oriented-design compared to C#. AT&T Asp.Net C++!!", set(), 100)
#test_post3 = Post(3, "", u"C++~$§%) is a :=; := :D :-)) ;-)))) testing is important! Blue houses are... ~ hehe wt~iop complicated programming-language compared to C#. AT&T Asp.Net C++ #1234 1234 !!", set(), 100)
##prepr.preprocess_posts([test_post1, test_post2, test_post3], tags, filter_untagged_posts=False, filter_less_relevant_posts=False)
##print "\n" + ("-"*80) + "\n" + str(test_post1.tokens) + "\n" + str(test_post2.tokens) + "\n" + str(test_post3.tokens) + "\n" + "-"*80
# DEBUG END
#new_post1 = Post(1, u"Do dynamic typed languages deserve all the criticism?", u"I have read a few articles on Internet about programming language choice in the enterprise. Recently many dynamic typed languages have been popular, i.e. Ruby, Python, PHP and Erlang. But many enterprises still stay with static typed languages like C, C++, C# and Java. And yes, one of the benefits of static typed languages is that programming errors are caught earlier, at compile time, rather than at run time. But there are also advantages with dynamic typed languages. (more on Wikipedia) The main reason why enterprises don't start to use languages like Erlang, Ruby and Python, seem to be the fact that they are dynamic typed. That also seem to be the main reason why people on StackOverflow decide against Erlang. See Why did you decide against Erlang. However, there seem to be a strong criticism against dynamic typing in the enterprises, but I don't really get it why it is that strong. Really, why is there so much criticism against dynamic typing in the enterprises? Does it really affect the cost of projects that much, or what? But maybe I'm wrong.", set())
#     new_post1 = Post(1, u"Java.util.List thread-safe?", u"Is a java.util.List thread-safe? From C++ I know that std::vectors are not thread-safe. Concurrency issues are very hard to debug and very important to find because nowadays most devices have a multicore cpu.", set(), 100)
#     new_post2 = Post(2, u"Choosing a Java Web Framework now?", u'we are in the planning stage of migrating a large website which is built on a custom developed mvc framework to a java based web framework which provides built-in support for ajax, rich media content, mashup, templates based layout, validation, maximum html/java code separation. Grails looked like a good choice, however, we do not want to use a scripting language. We want to continue using java. Template based layout is a primary concern as we intend to use this web application with multiple web sites with similar functionality but radically different look and feel. Is portal based solution a good fit to this problem? Any insights on using "Spring Roo" or "Play" will be very helpful. I did find similar posts like this, but it is more than a year old. Things have surely changed in the mean time! EDIT 1: Thanks for the great answers! This site is turning to be the best single source for in-the-trenches programmer info. However, I was expecting more info on using a portal-cms duo. Jahia looks goods. Anything similar?', set(), 100)
#     new_posts = prepr.preprocess_posts([new_post1, new_post2], tags, filter_untagged_posts=False, filter_less_relevant_posts=False)
#     print new_post2.tokens


def preprocess_tags_and_posts(all_tags, all_posts, tag_frequency_threshold):
    tags = prepr.filter_tags_and_sort_by_frequency(all_tags, tag_frequency_threshold)
    posts = prepr.preprocess_posts(all_posts, tags, filter_untagged_posts=True)
    Tag.update_tag_counts_according_to_posts(tags, posts)
    return tags, posts


def main():
    kwargs = usage()
    data_set_path = kwargs['data_set_path']
    enable_caching = kwargs['enable_caching']
    setup_logging(logging.INFO)
    helper.make_dir_if_not_exists(helper.CACHE_PATH)

    # 1) Parsing
    _logger.info("Parsing...")
    all_tags, all_posts, cache_file_name_prefix = parser.parse_tags_and_posts(data_set_path)
    all_posts_assignments = reduce(lambda x,y: x + y, map(lambda p: len(p.tag_set), all_posts))

    # 2) Preprocessing
    _logger.info("Preprocessing...")
    TAG_FREQUENCY_THRESHOLD = kwargs['tag_frequency_threshold']
    # TODO: FIXME: invalidate cache if tag-frequency has changed!!!
    if not enable_caching or not helper.cache_exists_for_preprocessed_tags_and_posts(cache_file_name_prefix):
        tags, posts = preprocess_tags_and_posts(all_tags, all_posts, TAG_FREQUENCY_THRESHOLD)
        if enable_caching:
            helper.write_preprocessed_tags_and_posts_to_cache(cache_file_name_prefix, tags, posts)
    else:
        _logger.info("Cache hit!")
        tags, posts = helper.load_preprocessed_tags_and_posts_from_cache(cache_file_name_prefix)

    helper.print_tags_summary(len(all_tags), len(tags))
    helper.print_posts_summary(all_posts, all_posts_assignments, posts)

    # 3) Split data set
    test_size = 0.1
    _logger.info("Splitting data set!")
    _logger.info(" Training: {}%, Test: {}%".format( (1-test_size)*100, test_size*100 ))
    # NOTE: last 2 return values are omitted since y-values are already
    #       included in our Post-instances
    train_posts, test_posts, _, _ = train_test_split(posts, np.zeros(len(posts)), test_size=test_size, random_state=42)

    # Suggest most frequent tags (baseline)
    _logger.info("-"*80)
    _logger.info("Randomly suggest 2 most frequent tags...")
    helper.suggest_random_tags(2, test_posts, tags)
    evaluation.print_evaluation_results(test_posts)
    _logger.info("-"*80)
    _logger.info("Only auggest most frequent tag '%s'..." % tags[0])
    helper.suggest_random_tags(1, test_posts, [tags[0]])
    evaluation.print_evaluation_results(test_posts)

    # 3) learning
    _logger.info("Learning...")

    #naive_bayes.naive_bayes_single_classifier(train_posts, test_posts, tags)
    _logger.info("-"*80)
    _logger.info("Naive bayes...")
    naive_bayes.naive_bayes(train_posts, test_posts, tags)
    evaluation.print_evaluation_results(test_posts)

    _logger.info("-"*80)
    _logger.info("k-Means...")
    kmeans.kmeans(len(tags), train_posts, test_posts)
    evaluation.print_evaluation_results(test_posts)

    # TODO: random forest...
    # TODO: linear SVM...

#     _logger.info("-"*80)
#     _logger.info("HAC...")
#     _logger.info("-"*80)
#     helper.clear_tag_predictions_for_posts(test_posts)
#     hac.hac(len(tags), train_posts, test_posts)
#     evaluation.print_evaluation_results(test_posts)
    return ExitCode.SUCCESS


if __name__ == "__main__":
    sys.exit(main())
