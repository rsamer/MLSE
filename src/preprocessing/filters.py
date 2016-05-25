# -*- coding: utf-8 -*-

from entities.post import Post
import logging
import nltk
import os
import re

main_dir = os.path.dirname(os.path.realpath(__file__)) + "/../../"
nltk.data.path = [main_dir + "corpora/nltk_data"]
emoticons_data_file = main_dir + "corpora/emoticons/emoticons"

tokens_punctuation_re = re.compile(r"(\.|!|\?|\(|\)|~)$")
single_character_tokens_re = re.compile(r"^\W$")

log = logging.getLogger("preprocessing.filters")


def to_lower_case(posts):
    logging.info("Lower case post title and body")
    for post in posts:
        post.title = post.title.lower()
        post.body = post.body.lower()


def strip_code_segments(posts):
    logging.info("Stripping code snippet from posts")
    for post in posts:
        assert (isinstance(post, Post))
        post.body = re.sub('<code>.*?</code>', '', post.body)


def strip_html_tags(posts):
    logging.info("Stripping HTML-tags from posts")
    try:
        from bs4 import BeautifulSoup  # @UnresolvedImport
    except ImportError:
        raise RuntimeError('Please install BeautifulSoup library!')

    for post in posts:
        assert (isinstance(post, Post))
        post.title = BeautifulSoup(post.title, "html.parser").text.strip()
        post.body = BeautifulSoup(post.body, "html.parser").text.strip()

def filter_tokens(posts, tag_names):
    logging.info("Filter posts' tokens")
    regex_url = re.compile(
        r'^(?:http|ftp)s?://'  # http:// https:// ftp:// ftps://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)

    # from nltk.tokenize.casual
    regex_emoticons = re.compile(r"""
        (?:
          [<>]?
          [:;=8]                     # eyes
          [\-o\*\']?                 # optional nose
          [\)\]\(\[dDpP/\:\}\{@\|\\] # mouth
          |
          [\)\]\(\[dDpP/\:\}\{@\|\\] # mouth
          [\-o\*\']?                 # optional nose
          [:;=8]                     # eyes
          [<>]?
          |
          <3                         # heart
        )""", re.IGNORECASE)

    regex_hex_numbers = re.compile(r'x[0-9a-fA-F]+', re.IGNORECASE)

    with open(emoticons_data_file) as emoticons_file:
        emoticons_list = emoticons_file.readlines()

    regex_number = re.compile(r'^#\d+$', re.IGNORECASE)

    total_number_of_filtered_tokens = 0
    total_number_of_tokens = 0
    for post in posts:
        tokens = post.tokens
        num_of_unfiltered_tokens = len(tokens)
        total_number_of_tokens += num_of_unfiltered_tokens

        # remove empty tokens, numbers and those single-character words that are no letters
        tokens = filter(lambda t: len(t) > 0 and not t.isdigit() and len(single_character_tokens_re.findall(t)) == 0,
                        tokens)

        # remove single-character words that are not part of our tag list
        tokens = filter(lambda t: len(t) > 1 or t in tag_names, tokens)

        # remove "'s" at the end
        # "'s" is not useful for us because it may only indicate the verb "be", "have"
        # (e.g. "He's got ..." or "It's ...")
        # or may indicate a possessive nouns (e.g. His wife's shoes...)
        tokens = map(lambda t: t[:-2] if t.endswith("'s") else t, tokens)

        # remove "s'" at the end
        # "s'" is also not useful for us since it indicates the plural version
        # of possessive nouns (e.g. the planets' orbits)
        tokens = map(lambda t: t[:-2] if t.endswith("s'") else t, tokens)

        # remove "'ve" at the end (e.g. we've got...)
        tokens = map(lambda t: t[:-2] if t.endswith("'ve") else t, tokens)

        # remove urls
        # XXX: not sure if this is removes important words! => word.startswith("https://")
        tokens = [word for word in tokens if regex_url.match(word) is None]

        # remove numbers starting with #
        tokens = [word for word in tokens if regex_number.match(word) is None]

        # remove hexadecimal numbers
        tokens = [word for word in tokens if regex_hex_numbers.match(word) is None]

        # remove emoticons (list from https://en.wikipedia.org/wiki/List_of_emoticons)
        tokens = [word for word in tokens if word not in emoticons_list]

        # remove emoticons (regex)
        tokens = [word for word in tokens if regex_emoticons.match(word) is None]

        post.tokens = tokens
        total_number_of_filtered_tokens += (num_of_unfiltered_tokens - len(post.tokens))

    if total_number_of_tokens != 0:
        logging.info("Removed {} ({}%) of {} tokens (altogether)".format(total_number_of_filtered_tokens,
                                                                         round(float(
                                                                             total_number_of_filtered_tokens) / total_number_of_tokens * 100.0,
                                                                               2),
                                                                         total_number_of_tokens))


def add_accepted_answer_text_to_body(posts):
    for post in posts:
        if post.accepted_answer_id is None:
            continue
        accepted_answer = post.accepted_answer()
        if accepted_answer is None:
            continue
        # assert accepted_answer is not None
        if accepted_answer.score >= 0:
            #                 print "-"*80
            #                 print accepted_answer.body
            post.body += " " + accepted_answer.body


def filter_less_relevant_posts(posts, score_threshold):
    logging.info("Filtering less relevant posts according to #answers and score value")
    # filter posts having low score according to given threshold
    posts = filter(lambda p: p.score >= score_threshold, posts)
    # posts = filter(lambda p: p.accepted_answer_id is not None, posts)
    posts = filter(lambda p: len(p.answers) > 0 or p.score > 0, posts)
    return posts
