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
@license:    license
'''

import sys
import os

from entities.tag import Tag
from entities.post import Post
from preprocessing import preprocessing as prepr
from unsupervised import clustering

TAG_FREQUENCY_THRESHOLD = 3

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

def main(argv=None):
    usage()
    path = sys.argv[1]

    tags_path = os.path.join(path, 'Tags.xml')
    posts_path = os.path.join(path, 'Posts.xml')

    tag_dict = Tag.parse_tags(tags_path) # 166,742 posts
    number_of_tags = len(tag_dict)
    sorted_filtered_tags = prepr.preprocess_tags_and_sort_by_frequency(map(lambda x: x[1], tag_dict.items()), TAG_FREQUENCY_THRESHOLD)

    print "Total number of tags: %d" % number_of_tags
    print "Number of remaining tags: %d" % len(sorted_filtered_tags)
    print "{}%".format(round(float(number_of_tags-len(sorted_filtered_tags))/number_of_tags*100.0, 2))

    posts = Post.parse_posts(posts_path, tag_dict)
    posts = prepr.preprocess_posts(posts, sorted_filtered_tags)
    # TODO: continue here...

    #new_post = Post(1, u"Do dynamic typed languages deserve all the criticism?", u"I have read a few articles on Internet about programming language choice in the enterprise. Recently many dynamic typed languages have been popular, i.e. Ruby, Python, PHP and Erlang. But many enterprises still stay with static typed languages like C, C++, C# and Java. And yes, one of the benefits of static typed languages is that programming errors are caught earlier, at compile time, rather than at run time. But there are also advantages with dynamic typed languages. (more on Wikipedia) The main reason why enterprises don't start to use languages like Erlang, Ruby and Python, seem to be the fact that they are dynamic typed. That also seem to be the main reason why people on StackOverflow decide against Erlang. See Why did you decide against Erlang. However, there seem to be a strong criticism against dynamic typing in the enterprises, but I don't really get it why it is that strong. Really, why is there so much criticism against dynamic typing in the enterprises? Does it really affect the cost of projects that much, or what? But maybe I'm wrong.", [])
    new_post = Post(1, u"Java.util.List thread-safe?", u"Is a java.util.List thread-safe? From C++ I know that std::vectors are not thread-safe. Concurrency issues are very hard to debug and very important to find because nowadays most devices have a multicore cpu.", [])
    prepr.preprocess_posts([new_post], sorted_filtered_tags)
    clustering.kmeans(len(posts) / 10, posts, new_post)

    return ErrorCode.SUCCESS

if __name__ == "__main__":
    sys.exit(main())
