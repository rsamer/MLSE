# -*- coding: utf-8 -*-

import logging
import nltk
import os

main_dir = os.path.dirname(os.path.realpath(__file__)) + "/../../"
nltk.data.path = [main_dir + "corpora/nltk_data"]

log = logging.getLogger("preprocessing.stemmer")


def porter_stemmer(posts):
    logging.info("Stemming for posts' tokens")
    try:
        from nltk.stem.porter import PorterStemmer
    except ImportError:
        raise RuntimeError('Please install nltk library!')

    porter = PorterStemmer()

    for post in posts:
        post.tokens = [porter.stem(word) for word in post.tokens]
