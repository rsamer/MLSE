# -*- coding: utf-8 -*-

import nltk
import os
import logging
from util import helper

_logger = logging.getLogger(__name__)
nltk.data.path = [os.path.join(helper.APP_PATH, "corpora", "nltk_data")]

def remove_stopwords(posts):
    _logger.info("Removing stop-words from posts' tokens")
    try:
        from nltk.corpus import stopwords
    except ImportError:
        raise RuntimeError('Please install nltk library!')

    additional_stop_words = ["e.g", "i.e", "vs", "vice-versa"] # without "." at the end!!
    stop_words = stopwords.words('english') + additional_stop_words

    progress_bar = helper.ProgressBar(len(posts))
    for post in posts:
        #post.tokens = [word for word in post.tokens if word not in stop_words]
        post.tokens = filter(lambda t: t not in stop_words, post.tokens)
        progress_bar.update()

    progress_bar.finish()
