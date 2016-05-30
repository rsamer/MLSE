# -*- coding: utf-8 -*-

import logging
from entities.post import Post

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
