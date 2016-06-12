# -*- coding: utf-8 -*-

import re
import logging
from util import helper

_logger = logging.getLogger(__name__)
tokens_punctuation_re = re.compile(r"(\.|,|'|-|!|:|;|\"|\?|/|\(|\)|~)$")
single_character_tokens_re = re.compile(r"^\W$")


def tokenize_posts(posts, tag_names):
    '''

        Customized tokenizer for our special needs.
        Unfortunately, many tokenizers out there are not aware of technical terms like C#, C++,
        asp.net, etc.

        Instead we could have used an existing tokenizer tool and:
         a) modify/adapt the tokenization-rules
         b) or use classification to train them
        but this would have had several drawbacks for us:
         a) the tokenizer would be still not flexible enough for our needs...
         b) we would have had to create much training data...

        Thus, we decided to come up with our own flexible tokenizer solution that is custom-designed
        to tokenize StackExchange posts efficiently.

        NOTE: In order to ensure that our tokenizer is working correct, we have created many
              unit-testcases for this python-module (see: "tests"-folder of this project)

    '''
    _logger.info("Tokenizing posts")
    assert isinstance(posts, list)
    assert isinstance(tag_names, list)

    tag_names = map(lambda n: n.lower(), tag_names)
    sorted_tag_names = sorted(tag_names, reverse=True)

    # based on: http://stackoverflow.com/a/36463112
    url_regex_str = r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+'
    simple_emoticons_regex_str = r'(?:\s[:;=\^\-oO][\-_\.]?[\)\(\]\[\-DPOp_\^\\\/]\s)'
    regex_str = [
        r"(?:[a-z][a-z'\-\.]+[a-z])",      # words containing "'", "-" or "."
        r'\w+\&\w+',                       # e.g. AT&T
        r'(?:@[\w_]+)',                    # @-mentions
        r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)",  # hash-tags
        # r'(?:\d+\%)',                    # percentage
        r'(?:(?:\d+,?)+(?:\.?\d+)?)',      # numbers
        r'(?:[\w_]+)',                     # other words
        r'(?:\S)'                          # anything else
    ]
    tokens_ignore_re = re.compile(r'(' + '|'.join(regex_str) + ')', re.VERBOSE | re.IGNORECASE)
    tokens_remove_url_re = re.compile(url_regex_str, re.VERBOSE | re.IGNORECASE)
    tokens_remove_emoticons_re = re.compile(simple_emoticons_regex_str, re.IGNORECASE)

    # regex to split/tokenize by tag name
    regex_str = map(lambda tag_name: '\s' + re.escape(tag_name) + '\W', sorted_tag_names)
    split_tag_re = re.compile(r'(' + '|'.join(regex_str) + ')', re.VERBOSE | re.IGNORECASE)
    last_char_is_no_alphanum_re = re.compile(r'(\W)', re.VERBOSE | re.IGNORECASE)

    def _tokenize_text(s, sorted_tag_names, split_tag_re):
        def _pre_tokenize_tag_parts(chunks, sorted_tag_names, split_tag_re):
            '''
                Recursively tokenizes text into tag and non-tag parts
            '''
            new_chunks = []
            for chunk in chunks:
                assert isinstance(chunk, (str, unicode))
                # case: chunk is a tag (i.e. chunk == tag_X)
                if chunk.strip() in sorted_tag_names:
                    new_chunks.append(chunk)
                    continue

                if chunk.strip()[:-1] in sorted_tag_names \
                and len(last_char_is_no_alphanum_re.findall(chunk.strip()[-1])) == 1:
                    new_chunks.append(chunk.strip()[:-1])
                    continue

                # case: chunk is not a tag
                # check if chunk contains (!) tag (i.e. tag_X is part of chunk!)
                sub_chunks = [chunk]
                sub_parts = split_tag_re.split(chunk)
                if len(sub_parts) > 1:
                    sub_parts = filter(lambda ch: len(ch.strip()) > 0, sub_parts)
                    sub_parts = map(lambda ch: ' '+ ch.strip() + ' ', sub_parts)
                    sub_chunks = _pre_tokenize_tag_parts(sub_parts, sorted_tag_names, split_tag_re)
                    sub_chunks = map(lambda ch: ch.strip(), sub_chunks)
                    sub_chunks = filter(lambda ch: len(ch) > 0, sub_chunks)
                new_chunks += sub_chunks
            return new_chunks


        def tokenize(s, sorted_tag_names, split_tag_re):
            tokens = _pre_tokenize_tag_parts([' %s ' % s], sorted_tag_names, split_tag_re)
            tokens = map(lambda t: t.strip(), tokens)
            final_tokens = []
            for token in tokens:
                assert isinstance(token, (str, unicode))
                # case: token is a tag -> do no further tokenization!
                if token in sorted_tag_names:
                    final_tokens.append(token)
                    continue

                # case: token is not a tag and even does not contain
                #       any tags. this token can be a single word or
                #       multiple words that are part of a sentence
                #       -> further tokenization required
                final_tokens += tokens_ignore_re.findall(' %s ' % token)
            return final_tokens


        def remove_all_tailing_punctuation_characters(token):
            old_token = token
            while True:
                if len(tokens_punctuation_re.findall(token)) == 0:
                    if len(old_token) >= 2 and old_token != token:
                        _logger.debug("Replaced: %s -> %s" % (old_token, token))
                    return token
                token = token[:-1]


        # pre- and append single whitespace character before and at the end of the string
        # This really makes the regular expressions a bit less complex
        s = ' ' + s.lower() + ' ' # also lower case all letters

        # remove URLs
        s = re.sub(tokens_remove_url_re, ' ', s)

        # remove simple emoticons (Note: the complexer ones are removed after tokenization later on!)
        s = re.sub(tokens_remove_emoticons_re, ' ', s)
        # TODO: detect additional smilies via regex...
        s = s.replace(':d', '')
        s = s.replace(':p', '')
        s = s.replace(':-d', '')
        s = s.replace(':-p', '')

        # split words by '_'
        s = ' '.join(s.split('_'))

        # tokenize
        tokens = tokenize(s, sorted_tag_names, split_tag_re)

        # remove whitespaces before and after
        tokens = map(lambda t: t.strip(), tokens)

        # remove tailing punctuation characters (if present)
        tokens = map(remove_all_tailing_punctuation_characters, tokens)

        # finally remove empty tokens
        return filter(lambda t: len(t) > 0, tokens)

    progress_bar = helper.ProgressBar(len(posts))
    for post in posts:
        post.title_tokens = _tokenize_text(post.title, sorted_tag_names, split_tag_re)
        post.body_tokens = _tokenize_text(post.body, sorted_tag_names, split_tag_re)
        progress_bar.update()
    progress_bar.finish()
