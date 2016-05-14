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
from unsupervised import clustering
import logging
import hashlib

SRC_PATH = os.path.realpath(os.path.dirname(__file__))
APP_PATH = os.path.join(SRC_PATH, "..")

class ErrorCode(object):
    SUCCESS = 0
    FAILED = 1


def is_complete_data_set(path):
    required_files = ["Posts.xml", "Tags.xml"]
    available_files = filter(lambda f: os.path.isfile(os.path.join(path, f)), required_files)
    return set(available_files) == set(required_files)


def usage():
    if len(sys.argv) != 2 or not is_complete_data_set(sys.argv[1]):
        print "No or invalid path given!"
        print "Usage: python -m main ../data/example/"
        sys.exit(ErrorCode.FAILED)


def print_tags_summary(total_num_of_tags, num_filtered_tags):
    print "-"*80
    print "Total number of tags: %d" % total_num_of_tags
    print "Number of remaining tags: %d" % num_filtered_tags
    print "Removed {}% of all tags".format(round(float(total_num_of_tags-num_filtered_tags)/total_num_of_tags*100.0, 2))
    print "-"*80


def print_posts_summary(all_posts, num_of_all_post_assignments, filtered_posts):
    print "-"*80
    print "Total number of posts: %d" % len(all_posts)
    print "Number of remaining posts: %d" % len(filtered_posts)
    print "Removed {}% of all posts".format(round(float(len(all_posts)-len(filtered_posts))/len(all_posts)*100.0, 2))

    new_posts_assignments = reduce(lambda x,y: x + y, map(lambda p: len(p.tag_set), filtered_posts))
    print "Removed {}% of all tag assignments".format(round(float(num_of_all_post_assignments-new_posts_assignments)/num_of_all_post_assignments*100.0, 2))
    print "-"*80


def setup_logging(log_level):
    logging.basicConfig(
        filename=None,
        level=log_level,
        format='%(asctime)s: %(levelname)7s: [%(name)s]: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )


def compute_hash_of_file(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


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
    all_posts_assignments = reduce(lambda x,y: x + y, map(lambda p: len(p.tag_set), all_posts))

    cache_dir = os.path.join(APP_PATH, "temp", "cache")
    cached_tags_file = os.path.join(cache_dir, full_hash + "_tags.pickle")
    cached_posts_file = os.path.join(cache_dir, full_hash + "_posts.pickle")
    if not os.path.exists(cached_tags_file) or not os.path.exists(cached_posts_file):
        TAG_FREQUENCY_THRESHOLD = 30#8
        tags = prepr.filter_tags_and_sort_by_frequency(all_tags, TAG_FREQUENCY_THRESHOLD)
        posts = prepr.preprocess_posts(all_posts, tags, filter_untagged_posts=True)
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

    print_tags_summary(len(all_tags), len(tags))
    print_posts_summary(all_posts, all_posts_assignments, posts)

    # DEBUG BEGIN
    test_post1 = Post(1, "", u"RT @marcobonzanini: just, an example! :D http://example.com/what?q=test #NLP", set())
    test_post2 = Post(2, "", u"A b C d e f g h i j k f# u# and C++ is a test hehe wt iop complicated programming-languages object oriented object-oriented-design compared to C#. AT&T Asp.Net C++!!", set())
    test_post3 = Post(3, "", u"C++~$§%) is a :=; := :D :-)) ;-)))) testing is important! Blue houses are... ~ hehe wt~iop complicated programming-language compared to C#. AT&T Asp.Net C++ #1234 1234 !!", set())
    prepr.preprocess_posts([test_post1, test_post2, test_post3], tags, filter_untagged_posts=False)
    print "\n" + ("-"*80) + "\n" + str(test_post1.tokens) + "\n" + str(test_post2.tokens) + "\n" + str(test_post3.tokens) + "\n" + "-"*80
    # DEBUG END

    logging.info("Finished pre-processing!")

    # learning
    #new_post = Post(1, u"Do dynamic typed languages deserve all the criticism?", u"I have read a few articles on Internet about programming language choice in the enterprise. Recently many dynamic typed languages have been popular, i.e. Ruby, Python, PHP and Erlang. But many enterprises still stay with static typed languages like C, C++, C# and Java. And yes, one of the benefits of static typed languages is that programming errors are caught earlier, at compile time, rather than at run time. But there are also advantages with dynamic typed languages. (more on Wikipedia) The main reason why enterprises don't start to use languages like Erlang, Ruby and Python, seem to be the fact that they are dynamic typed. That also seem to be the main reason why people on StackOverflow decide against Erlang. See Why did you decide against Erlang. However, there seem to be a strong criticism against dynamic typing in the enterprises, but I don't really get it why it is that strong. Really, why is there so much criticism against dynamic typing in the enterprises? Does it really affect the cost of projects that much, or what? But maybe I'm wrong.", set())
    new_post = Post(1, u"Java.util.List thread-safe?", u"Is a java.util.List thread-safe? From C++ I know that std::vectors are not thread-safe. Concurrency issues are very hard to debug and very important to find because nowadays most devices have a multicore cpu.", set())
    [new_post] = prepr.preprocess_posts([new_post], tags, filter_untagged_posts=False)
    clustering.kmeans(len(posts) / 10, posts, new_post)

    # TODO: continue here...

    return ErrorCode.SUCCESS

if __name__ == "__main__":
    sys.exit(main())
