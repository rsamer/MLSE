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

    data_set_stop_words = [u'able', u'across', u'actual', u'actually', u'add', u'adding', u'allow', u'almost', u'already', u'also', u'always', u'amount', u'another', u'anyone', u'anything', u'appropriate', u'around', u'available', u'avoid', u'away', u'back', u'bad', u'base', u'based', u'basic', u'basically', u'become', u'believe', u'benefits', u'best', u'better', u'big', u'bit', u'break', u'call', u'called', u'cannot', u'care', u'case', u'cases', u'certain', u'change', u'changes', u'check', u'choice', u'choose', u'clear', u'come', u'comes', u'common', u'complete', u'completely', u'complex', u'consider', u'considered', u'correct', u'could', u'cross', u'current', u'currently', u'day', u'days', u'decide', u'details', u'difference', u'different', u'difficult', u'directly', u'done', u'easier', u'easily', u'easy', u'either', u'else', u'end', u'enough', u'especially', u'etc', u'even', u'ever', u'every', u'everyone', u'everything', u'exactly', u'example', u'examples', u'existing', u'expect', u'far', u'feel', u'first', u'free', u'full', u'general', u'generally', u'get', u'getting', u'going', u'good', u'got', u'great', u'hand', u'hard', u'help', u'high', u'hours', u'however', u'include', u'instead', u'interested', u'interesting', u'involved', u'keep', u'kind', u'know', u'large', u'last', u'later', u'like', u'likely', u'little', u'local', u'look', u'looking', u'lot', u'low', u'made', u'major', u'many', u'may', u'maybe', u'mean', u'might', u'mostly', u'much', u'multiple', u'must', u'necessary', u'need', u'needed', u'never', u'next', u'nothing', u'often', u'old', u'one', u'ones', u'others', u'particular', u'per', u'please', u'possible', u'pretty', u'probably', u'quickly', u'quite', u'rather', u'real', u'really', u'recently', u'related', u'said', u'say', u'second', u'see', u'seem', u'seems', u'seen', u'several', u'short', u'similar', u'simple', u'simply', u'since', u'small', u'someone', u'something', u'sometimes', u'specific', u'still', u'sure', u'thanks', u'though', u'thought', u'three', u'times', u'two', u'use', u'used', u'useful', u'using', u'usually', u'various', u'want', u'way', u'ways', u'well', u'whatever', u'whether', u'whole', u'within', u'without', u'wondering', u'would', u'year', u'years', u'yes', u'yet']
    additional_stop_words = ["e.g", "i.e", "vs", "vice-versa"] + data_set_stop_words # without "." at the end!!
    stop_words = stopwords.words('english') + additional_stop_words

    def _remove_stopwords(tokens):
        return filter(lambda t: t not in stop_words, tokens)

    progress_bar = helper.ProgressBar(len(posts))
    for post in posts:
        post.title_tokens = _remove_stopwords(post.title_tokens)
        post.body_tokens = _remove_stopwords(post.body_tokens)
        progress_bar.update()

    progress_bar.finish()
