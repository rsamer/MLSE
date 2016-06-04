# -*- coding: utf-8 -*-

import csv, logging, os
from entities.tag import Tag
from entities.post import Post
from util import helper

_logger = logging.getLogger(__name__)


def replace_tag_synonyms(tags, posts):
    synonyms_file_path = os.path.join(helper.APP_PATH, 'corpora', 'tags', 'synonyms')
    tag_name_replacement_map = {}
    with open(synonyms_file_path, 'rb') as f:
        reader = csv.reader(f)
        for row in reader:
            assert row[1] not in tag_name_replacement_map
            tag_name_replacement_map[row[1]] = row[0]

    #remaining_tags = filter(lambda tag: tag.name not in tag_replacements, tags)
    tag_name_tag_map = dict(map(lambda t: (t.name, t), tags))
    remaining_tags = filter(lambda t: t.name not in tag_name_replacement_map, tags)
    #print tags
    counter_assigned_synonym_tags = 0
    for post in posts:
        assert isinstance(post, Post)
        new_tag_set = set()
        for tag in post.tag_set:
            if tag.name in tag_name_replacement_map:
                new_tag_name = tag_name_replacement_map[tag.name]
                if new_tag_name not in tag_name_tag_map:
                    _logger.debug("Ignoring non existant tag %s", new_tag_name)
                    continue
                _logger.debug("Replaced %s with %s", tag, tag_name_tag_map[new_tag_name])
                new_tag_set.add(tag_name_tag_map[new_tag_name])
                counter_assigned_synonym_tags += 1
            else:
                new_tag_set.add(tag)
        post.tag_set = new_tag_set
        assert len(post.tag_set) > 0
    _logger.info("Found and replaced %s synonym tags", len(tags) - len(remaining_tags))
    _logger.info("Replaced %s assignments of synonym tags in all posts", counter_assigned_synonym_tags)
    Tag.update_tag_counts_according_to_posts(remaining_tags, posts)
#     sort_tags = sorted(remaining_tags, key=lambda x: x.count, reverse=True)
#     print "-"*80
#     for t in sort_tags:
#         if t.count < 6:
#             break
#         print t
#     print "-"*80
#     import sys;sys.exit()
    return remaining_tags, posts


def replace_adjacent_tag_occurences(posts, tag_names):
    ''' replaces "-" by " " in all tag names e.g. "object-oriented" -> "object oriented"
        and then looks for two (or more) adjacent words that represent a known tag name
        e.g. current token list ["I", "love", "object", "oriented", "code"]
        -> should be converted to ["I", "love", "object-oriented", "code"]
        since "object-oriented" is a tag name in our tag list
    '''
    # TODO: if body contains text as tag
    for tag_name in tag_names:
        if "-" not in tag_name:
            continue

        splitted_tag_name = tag_name.replace("-", " ")
        for post in posts:
            assert isinstance(post, Post)
            post.title = post.title.replace(splitted_tag_name, tag_name)
            post.body = post.body.replace(splitted_tag_name, tag_name)


def strip_invalid_tags_from_posts_and_remove_untagged_posts(posts, tags):
    ''' unassigns all removed tags from posts to avoid data-inconsistency issues '''
    _logger.info("Stripping invalid tags from posts and removing untagged posts")
    new_post_list = []
    for post in posts:
        assert isinstance(post, Post)
        post.tag_set = post.tag_set.intersection(tags) # removes invalid tags
        if len(post.tag_set) > 0: # removes untagged posts
            new_post_list.append(post)
    return new_post_list
