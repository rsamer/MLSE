# -*- coding: utf-8 -*-

import logging

_logger = logging.getLogger(__name__)

def replace_adjacent_tag_occurences(posts, tag_names):
    ''' replaces "-" by " " in all tag names e.g. "object-oriented" -> "object oriented"
        and then looks for two (or more) adjacent words that represent a known tag name
        e.g. current token list ["I", "love", "object", "oriented", "code"]
        -> should be converted to ["I", "love", "object-oriented", "code"]
        since "object-oriented" is a tag name in our tag list
    '''
    # TODO: if body contains text as tag
    for tag_name in tag_names:
        splitted_tag_name = tag_name.replace("-", " ")
        if splitted_tag_name == tag_name:
            continue
        for post in posts:
            post.body = post.body.replace(splitted_tag_name, tag_name)


def strip_invalid_tags_from_posts_and_remove_untagged_posts(posts, tags):
    ''' unassigns all removed tags from posts to avoid data-inconsistency issues '''
    _logger.info("Stripping invalid tags from posts and removing untagged posts")
    new_post_list = []
    for post in posts:
        post.tag_set = post.tag_set.intersection(tags)
        if len(post.tag_set) > 0:
            new_post_list.append(post)
    return new_post_list
