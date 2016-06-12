# -*- coding: utf-8 -*-

import logging

_logger = logging.getLogger(__name__)


def append_accepted_answer_text_to_body(posts):
    assert isinstance(posts, list)

    for post in posts:
        accepted_answer = post.accepted_answer()
        if accepted_answer is None:
            continue
        if accepted_answer.score >= 0: # do not include negatively rated answers!
            post.body += " " + accepted_answer.body
