# -*- coding: utf-8 -*-

import nltk
import os
import logging
from util import helper
from nltk.corpus import stopwords

_logger = logging.getLogger(__name__)
nltk.data.path = [os.path.join(helper.APP_PATH, "corpora", "nltk_data")]


def remove_stopwords(posts):
    _logger.info("Removing stop-words from posts' tokens")
    stop_words_file_path = os.path.join(helper.APP_PATH, 'corpora', 'stopwords')
    data_set_stop_words = set()
    with open(stop_words_file_path, 'rb') as f:
        for line in f:
            data_set_stop_words.add(line.strip())

    stop_words = stopwords.words('english') + list(data_set_stop_words)

    progress_bar = helper.ProgressBar(len(posts))
    for post in posts:
        post.title_tokens = filter(lambda t: t not in stop_words, post.title_tokens)
        post.body_tokens = filter(lambda t: t not in stop_words, post.body_tokens)
        progress_bar.update()
    progress_bar.finish()
