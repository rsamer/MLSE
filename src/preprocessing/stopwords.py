# -*- coding: utf-8 -*-

import nltk
import os
import logging

_logger = logging.getLogger(__name__)
main_dir = os.path.dirname(os.path.realpath(__file__)) + "/../../"
nltk.data.path = [main_dir + "corpora/nltk_data"]

def remove_stopwords(posts):
    _logger.info("Removing stop-words from posts' tokens")
    try:
        from nltk.corpus import stopwords
    except ImportError:
        raise RuntimeError('Please install nltk library!')

    stop_words = stopwords.words('english')

    for post in posts:
        post.tokens = [word for word in post.tokens if word not in stop_words]
