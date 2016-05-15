# -*- coding: utf-8 -*-

import os
import sys
import hashlib


SRC_PATH = os.path.join(os.path.realpath(os.path.dirname(__file__)), "..")
APP_PATH = os.path.join(SRC_PATH, "..")


class ExitCode(object):
    SUCCESS = 0
    FAILED = 1


def make_dir_if_not_exists(path):
    paths = [path] if type(path) is not list else path
    for path in paths:
        try:
            if not os.path.exists(path):
                os.makedirs(path)
            elif not os.path.isdir(path):
                error("Invalid path '{0}'. This is NO directory.".format(path))
        except Exception, e:
            error(e)


def error(msg):
    print("ERROR: {0}".format(msg))
    sys.exit(ExitCode.FAILED)


def is_complete_data_set(path):
    required_files = ["Posts.xml", "Tags.xml"]
    available_files = filter(lambda f: os.path.isfile(os.path.join(path, f)), required_files)
    return set(available_files) == set(required_files)


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


def compute_hash_of_file(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

