# -*- coding: utf-8 -*-

import logging
import nltk
import os
from util import helper
from nltk.stem.porter import PorterStemmer

_logger = logging.getLogger(__name__)
nltk.data.path = [os.path.join(helper.APP_PATH, "corpora", "nltk_data")]


def porter_stemmer_tags(tags):
    _logger.info("Stemming tags")
    porter = PorterStemmer()
    progress_bar = helper.ProgressBar(len(tags))
    for tag in tags:
        tag.preprocessed_tag_name = porter.stem(tag.name.lower())
        assert len(tag.preprocessed_tag_name) > 0
        progress_bar.update()
    progress_bar.finish()


def porter_stemmer(posts):
    _logger.info("Stemming for posts' tokens")
    porter = PorterStemmer()

    progress_bar = helper.ProgressBar(len(posts))
    for post in posts:
        post.title_tokens = [porter.stem(word) for word in post.title_tokens]
        post.body_tokens = [porter.stem(word) for word in post.body_tokens]
        progress_bar.update()
    progress_bar.finish()
