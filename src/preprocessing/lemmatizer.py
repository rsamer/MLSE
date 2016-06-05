# -*- coding: utf-8 -*-

import logging
import nltk
from nltk.corpus import wordnet
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

    def map_stanford_to_wordnet_tag(stanford_tag):
        if stanford_tag.startswith('JJ'):
            return wordnet.ADJ
        elif stanford_tag.startswith('VB'):
            return wordnet.VERB
        elif stanford_tag.startswith('NN'):
            return wordnet.NOUN
        elif stanford_tag.startswith('RB'):
            return wordnet.ADV
        else:
            return ''

    lemmatizer = WordNetLemmatizer()

    for post in posts:
        post.title_tokens = [lemmatizer.lemmatize(pos_tag[0], map_stanford_to_wordnet_tag(pos_tag[1])) for pos_tag in post.title_tokens_pos_tags]
        post.body_tokens = [lemmatizer.lemmatize(pos_tag[0], map_stanford_to_wordnet_tag(pos_tag[1])) for pos_tag in post.body_tokens_pos_tags]
