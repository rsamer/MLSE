# -*- coding: utf-8 -*-

import re
import logging
from util import helper

_logger = logging.getLogger(__name__)
tokens_punctuation_re = re.compile(r"(\.|,|'|!|:|\"|\?|/|\(|\)|~)$")
single_character_tokens_re = re.compile(r"^\W$")


def tokenize_posts(posts, tag_names):
    ''' Customized tokenizer for our special needs! (e.g. C#, C++, ...) '''
    _logger.info("Tokenizing posts")
    # based on: http://stackoverflow.com/a/36463112
    regex_str = [
        # r'(?:[:;=\^\-oO][\-_\.]?[\)\(\]\[\-DPOp_\^\\\/])', # emoticons (Note: they are removed after tokenization!) # TODO why not here?
        r'\w+\.\w+',
        r'\w+\&\w+',  # e.g. AT&T
        r'(?:@[\w_]+)',  # @-mentions
        r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)",  # hash-tags
        r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+',
        # URLs  (Note: they are removed after tokenization!)
        # r'(?:\d+\%)', # percentage
        r'(?:(?:\d+,?)+(?:\.?\d+)?)',  # numbers
        r"(?:[a-z][a-z'\-_]+[a-z])",  # words with - and '
    ] + map(lambda tag_name: re.escape(tag_name) + '\W', tag_names) + [  # consider known tag names
        r'(?:[\w_]+)',  # other words
        r'(?:\S)'  # anything else
    ]
    tokens_ignore_re = re.compile(r'(' + '|'.join(regex_str) + ')', re.VERBOSE | re.IGNORECASE)

    def _tokenize_text(s, tag_names):
        def tokenize(s):
            return tokens_ignore_re.findall(s)

        def filter_all_tailing_punctuation_characters(token):
            old_token = token
            while True:
                punctuation_character = tokens_punctuation_re.findall(token)
                if len(punctuation_character) == 0:
                    if len(old_token) >= 2 and old_token != token:
                        _logger.debug("Replaced: %s -> %s" % (old_token, token))
                    return token
                token = token[:-1]

        # prepending and appending a single whitespace to the text
        # makes the regular expressions less complicated
        tokens = tokenize(" " + s + " ")
        tokens = [token.strip() for token in tokens]  # remove whitespaces before and after
        tokens = map(filter_all_tailing_punctuation_characters, tokens)
        return tokens

    progress_bar = helper.ProgressBar(len(posts))
    for post in posts:
        text = ((post.title + " ") * 10) + post.body
        post.tokens = _tokenize_text(text, tag_names)
        progress_bar.update()

    progress_bar.finish()

