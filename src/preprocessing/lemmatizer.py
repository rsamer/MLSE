# -*- coding: utf-8 -*-

import logging
import nltk
import os
from util import helper

_logger = logging.getLogger(__name__)
nltk.data.path = [os.path.join(helper.APP_PATH, "corpora", "nltk_data")]


def word_net_lemmatizer(posts):
    _logger.info("Lemmatization for posts' tokens")
    try:
        from nltk.stem.wordnet import WordNetLemmatizer
    except ImportError:
        raise RuntimeError('Please install nltk library!')

    lemmatizer = WordNetLemmatizer()

    for post in posts:
        post.tokens = [lemmatizer.lemmatize(word) for word in post.tokens]
