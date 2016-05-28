# -*- coding: utf-8 -*-

import logging
import nltk
import os

_logger = logging.getLogger(__name__)
main_dir = os.path.dirname(os.path.realpath(__file__)) + "/../../"
nltk.data.path = [main_dir + "corpora/nltk_data"]

def porter_stemmer(posts):
    _logger.info("Stemming for posts' tokens")
    try:
        from nltk.stem.porter import PorterStemmer
    except ImportError:
        raise RuntimeError('Please install nltk library!')

    porter = PorterStemmer()

    for post in posts:
        post.tokens = [porter.stem(word) for word in post.tokens]
