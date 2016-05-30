# -*- coding: utf-8 -*-

from entities.post import Post
import logging, re, os

_logger = logging.getLogger(__name__)


def add_accepted_answer_text_to_body(posts):
    assert isinstance(posts, list)

    for post in posts:
        accepted_answer = post.accepted_answer()
        if accepted_answer is None:
            continue
        # assert accepted_answer is not None
        if accepted_answer.score >= 0: # do not include negatively rated answers!
            post.body += " " + accepted_answer.body
#             print "-"*80
#             print accepted_answer.body


def add_title_to_body(posts):
    assert isinstance(posts, list)

    for post in posts:
        post.body = ((post.title + " ") * 10) + post.body