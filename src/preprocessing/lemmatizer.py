# -*- coding: utf-8 -*-

import logging
import nltk
import os

_logger = logging.getLogger(__name__)
main_dir = os.path.dirname(os.path.realpath(__file__)) + "/../../"
nltk.data.path = [main_dir + "corpora/nltk_data"]


def word_net_lemmatizer(posts):
    _logger.info("Lemmatization for posts' tokens")
    try:
        from nltk.stem.wordnet import WordNetLemmatizer
    except ImportError:
        raise RuntimeError('Please install nltk library!')

    lemmatizer = WordNetLemmatizer()

    for post in posts:
        post.tokens = [lemmatizer.lemmatize(word) for word in post.tokens]
