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

class ErrorCode(object):
    SUCCESS = 0
    FAILED = 1

TAG_FREQUENCY_THRESHOLD = 3

def main(argv=None):
    if len(sys.argv) != 2:
        print "Invalid number of arguments given!"
        print "Usage: python -m main ../data/example/"
        return ErrorCode.FAILED

    path = sys.argv[1]
    tags_path = os.path.join(path, 'Tags.xml')
    posts_path = os.path.join(path, 'Posts.xml')

    tag_dict = Tag.parse_tags(tags_path) # 166,742 posts
    number_of_tags = len(tag_dict)
    sorted_filtered_tags = prepr.preprocess_tags_and_sort_by_frequency(tag_dict.items(), TAG_FREQUENCY_THRESHOLD)

    print "Total number of tags: %d" % number_of_tags
    print "Number of remaining tags: %d" % len(sorted_filtered_tags)
    print "{}%".format(round(float(number_of_tags-len(sorted_filtered_tags))/number_of_tags*100.0, 2))
    print sorted_filtered_tags[-10:]

    posts = Post.parse_posts(posts_path, tag_dict)
    filtered_posts = prepr.preprocess_posts(posts)
    print "-"*80
    print "\n".join([post.body for post in filtered_posts[3:6]])
    return ErrorCode.SUCCESS

if __name__ == "__main__":
    sys.exit(main())
