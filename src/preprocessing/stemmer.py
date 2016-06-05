# -*- coding: utf-8 -*-

import logging
import nltk
import os
from util import helper

_logger = logging.getLogger(__name__)
nltk.data.path = [os.path.join(helper.APP_PATH, "corpora", "nltk_data")]


def porter_stemmer_tags(tags):
    _logger.info("Stemming tags")
    try:
        from nltk.stem.porter import PorterStemmer
    except ImportError:
        raise RuntimeError('Please install nltk library!')

    porter = PorterStemmer()

    progress_bar = helper.ProgressBar(len(tags))
    for tag in tags:
        tag.preprocessed_tag_name = porter.stem(tag.name.lower())
        progress_bar.update()

    progress_bar.finish()


def porter_stemmer(posts):
    _logger.info("Stemming for posts' tokens")
    try:
        from nltk.stem.porter import PorterStemmer
    except ImportError:
        raise RuntimeError('Please install nltk library!')

    porter = PorterStemmer()

    progress_bar = helper.ProgressBar(len(posts))
    for post in posts:
        post.body_tokens = [porter.stem(word) for word in post.body_tokens]
        post.title_tokens = [porter.stem(word) for word in post.title_tokens]
        progress_bar.update()

    progress_bar.finish()
