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

import sys
import os
import pickle

from entities import tag
from entities import post
from entities.tag import Tag
from entities.post import Post
from preprocessing import preprocessing as prepr
from unsupervised import evaluation, hac
from unsupervised import kmeans
from util.helper import ExitCode, error, compute_hash_of_file
from util import helper
import logging
from sklearn.cross_validation import train_test_split
import numpy as np


def usage():
    if len(sys.argv) != 2 or not helper.is_complete_data_set(sys.argv[1]):
        error("No or invalid path given!\nUsage: python -m main ../data/example/")


def setup_logging(log_level):
    logging.basicConfig(
        filename=None,
        level=log_level,
        format='%(asctime)s: %(levelname)7s: [%(name)s]: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )


def main(argv=None):
    usage()
    path = sys.argv[1]
    setup_logging(logging.DEBUG)

    # preprocessing
    tags_path = os.path.join(path, 'Tags.xml')
    posts_path = os.path.join(path, 'Posts.xml')
    full_hash = compute_hash_of_file(tags_path) + "_" + compute_hash_of_file(posts_path) + "_" \
              + compute_hash_of_file(prepr.__file__) + "_" \
              + compute_hash_of_file(tag.__file__) + "_" + compute_hash_of_file(post.__file__)

    name_to_tag_map = Tag.parse_tags(tags_path)
    all_tags = name_to_tag_map.values()
    all_posts = Post.parse_posts(posts_path, name_to_tag_map)
    Tag.update_tag_counts_according_to_posts(all_tags, all_posts)
    all_posts_assignments = reduce(lambda x,y: x + y, map(lambda p: len(p.tag_set), all_posts))

    cache_dir = os.path.join(helper.APP_PATH, "temp", "cache")
    helper.make_dir_if_not_exists(cache_dir)
    cached_tags_file = os.path.join(cache_dir, full_hash + "_tags.pickle")
    cached_posts_file = os.path.join(cache_dir, full_hash + "_posts.pickle")
    if not os.path.exists(cached_tags_file) or not os.path.exists(cached_posts_file):
        TAG_FREQUENCY_THRESHOLD = 8
        tags = prepr.filter_tags_and_sort_by_frequency(all_tags, TAG_FREQUENCY_THRESHOLD)
        posts = prepr.preprocess_posts(all_posts, tags, filter_untagged_posts=True)
        Tag.update_tag_counts_according_to_posts(tags, posts)
        with open(cached_tags_file, 'wb') as fp:
            pickle.dump(tags, fp)
        with open(cached_posts_file, 'wb') as fp:
            pickle.dump(posts, fp)
    else:
        # loaded from cache
        logging.info("Cache hit!")
        with open(cached_tags_file, 'r') as fp:
            tags = pickle.load(fp)
        with open(cached_posts_file, 'r') as fp:
            posts = pickle.load(fp)

    helper.print_tags_summary(len(all_tags), len(tags))
    helper.print_posts_summary(all_posts, all_posts_assignments, posts)

    # DEBUG BEGIN
    test_post1 = Post(1, "", u"RT @marcobonzanini: just, an example! :D http://example.com/what?q=test #NLP", set(), 100)
    test_post2 = Post(2, "", u"0x2AF3 #143152 A b C d e f g h i j k f# u# and C++ is a test hehe wt iop complicated programming-languages object oriented object-oriented-design compared to C#. AT&T Asp.Net C++!!", set(), 100)
    test_post3 = Post(3, "", u"C++~$§%) is a :=; := :D :-)) ;-)))) testing is important! Blue houses are... ~ hehe wt~iop complicated programming-language compared to C#. AT&T Asp.Net C++ #1234 1234 !!", set(), 100)
    prepr.preprocess_posts([test_post1, test_post2, test_post3], tags, filter_untagged_posts=False, filter_less_relevant_posts=False)
    print "\n" + ("-"*80) + "\n" + str(test_post1.tokens) + "\n" + str(test_post2.tokens) + "\n" + str(test_post3.tokens) + "\n" + "-"*80
    # DEBUG END

    logging.info("Finished pre-processing!")

    # learning
    #new_post1 = Post(1, u"Do dynamic typed languages deserve all the criticism?", u"I have read a few articles on Internet about programming language choice in the enterprise. Recently many dynamic typed languages have been popular, i.e. Ruby, Python, PHP and Erlang. But many enterprises still stay with static typed languages like C, C++, C# and Java. And yes, one of the benefits of static typed languages is that programming errors are caught earlier, at compile time, rather than at run time. But there are also advantages with dynamic typed languages. (more on Wikipedia) The main reason why enterprises don't start to use languages like Erlang, Ruby and Python, seem to be the fact that they are dynamic typed. That also seem to be the main reason why people on StackOverflow decide against Erlang. See Why did you decide against Erlang. However, there seem to be a strong criticism against dynamic typing in the enterprises, but I don't really get it why it is that strong. Really, why is there so much criticism against dynamic typing in the enterprises? Does it really affect the cost of projects that much, or what? But maybe I'm wrong.", set())
#     new_post1 = Post(1, u"Java.util.List thread-safe?", u"Is a java.util.List thread-safe? From C++ I know that std::vectors are not thread-safe. Concurrency issues are very hard to debug and very important to find because nowadays most devices have a multicore cpu.", set(), 100)
#     new_post2 = Post(2, u"Choosing a Java Web Framework now?", u'we are in the planning stage of migrating a large website which is built on a custom developed mvc framework to a java based web framework which provides built-in support for ajax, rich media content, mashup, templates based layout, validation, maximum html/java code separation. Grails looked like a good choice, however, we do not want to use a scripting language. We want to continue using java. Template based layout is a primary concern as we intend to use this web application with multiple web sites with similar functionality but radically different look and feel. Is portal based solution a good fit to this problem? Any insights on using "Spring Roo" or "Play" will be very helpful. I did find similar posts like this, but it is more than a year old. Things have surely changed in the mean time! EDIT 1: Thanks for the great answers! This site is turning to be the best single source for in-the-trenches programmer info. However, I was expecting more info on using a portal-cms duo. Jahia looks goods. Anything similar?', set(), 100)
#     new_posts = prepr.preprocess_posts([new_post1, new_post2], tags, filter_untagged_posts=False, filter_less_relevant_posts=False)
#     print new_post2.tokens
#     result = kmeans.kmeans(len(tags)/3, posts, new_posts)
#     print result


    # split data set
    train_posts, test_posts, _, _ = train_test_split(posts, np.zeros(len(posts)), test_size=0.1, random_state=42)

    from supervised import naive_bayes
    #naive_bayes.naive_bayes_single_classifier(train_posts, test_posts, tags)
    naive_bayes.naive_bayes(train_posts, test_posts, tags)
    precision = evaluation.precision(test_posts)
    recall = evaluation.recall(test_posts)
    print "Overall precision = " + str(precision)
    print "Overall recall = " + str(recall)
    sys.exit()

    print "-" * 80
    print "k-Means kmeans"
    print "-" * 80
    kmeans.kmeans(len(tags), train_posts, test_posts)
    # evaluation of kmeans
    precision = evaluation.precision(test_posts)
    recall = evaluation.recall(test_posts)
    print "Overall precision = " + str(precision)
    print "Overall recall = " + str(recall)

#     print "-" * 80
#     print "HAC kmeans"
#     print "-" * 80
#     helper.clear_tag_predictions_for_posts(test_posts)
#     hac.hac(len(tags), train_posts, test_posts)
#     # evaluation of hac
#     precision = evaluation.precision(test_posts)
#     recall = evaluation.recall(test_posts)
#     print "Overall precision = " + str(precision)
#     print "Overall recall = " + str(recall)

    return ExitCode.SUCCESS

if __name__ == "__main__":
    sys.exit(main())
