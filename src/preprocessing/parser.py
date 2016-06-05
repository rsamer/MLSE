# -*- coding: utf-8 -*-

import os
from entities.tag import Tag
from entities.post import Post
from util.helper import compute_hash_of_file


def parse_tags_and_posts(path):
    tags_path, posts_path = os.path.join(path, 'Tags.xml'), os.path.join(path, 'Posts.xml')
    cache_file_name_prefix = compute_hash_of_file(tags_path) + "_" + compute_hash_of_file(posts_path)
    tag_name_to_tag_map = Tag.parse_tags(tags_path)
    all_tags = tag_name_to_tag_map.values()
    all_posts = Post.parse_posts(posts_path, tag_name_to_tag_map)
    Tag.update_tag_counts_according_to_posts(all_tags, all_posts)
    return all_tags, all_posts, cache_file_name_prefix
