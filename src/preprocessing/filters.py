# -*- coding: utf-8 -*-

from entities.post import Post
import logging, re, os
from util import helper

_logger = logging.getLogger(__name__)
emoticons_data_file = os.path.join(helper.APP_PATH, "corpora", "emoticons", "emoticons")

tokens_punctuation_re = re.compile(r"(\.|!|\?|\(|\)|~)$")
single_character_tokens_re = re.compile(r"^\W$")

KNOWN_FILE_EXTENSIONS_MAP = {
    "exe": "windows",
    "jar": "java",
    "js": "javascript",
    "h": "c",
    "py": "python",
    "s": "assembly",
    "rb": "ruby"
}


def to_lower_case(posts):
    _logger.info("Lower case post title and body")
    for post in posts:
        assert isinstance(post, Post)
        post.title = post.title.lower()
        post.body = post.body.lower()


def strip_code_segments(posts):
    _logger.info("Stripping code snippet from posts")
    for post in posts:
        assert isinstance(post, Post)
        post.body = re.sub('\s*<code>.*?</code>\s*', ' ', post.body)


def strip_html_tags(posts):
    _logger.info("Stripping HTML-tags from posts")
    try:
        from bs4 import BeautifulSoup  # @UnresolvedImport
    except ImportError:
        raise RuntimeError('Please install BeautifulSoup library!')

    for post in posts:
        assert isinstance(post, Post)
        post.title = post.title.replace("&nbsp;", " ")
        post.body = post.body.replace("&nbsp;", " ")

        post.title = BeautifulSoup(post.title, "html.parser").text.strip()
        post.body = BeautifulSoup(post.body, "html.parser").text.strip()


def filter_tokens(posts, tag_names):
    _logger.info("Filter posts' tokens")
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

    regex_hex_numbers = re.compile(r'^0?x[0-9a-fA-F]+$', re.IGNORECASE)
    regex_number = re.compile(r'^#\d+$', re.IGNORECASE)
    regex_color_code = re.compile(r'^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})$', re.IGNORECASE)
    regex_float_number = re.compile(r'^\d+\.\d+$', re.IGNORECASE)
    regex_long_number_in_separated_format = re.compile(r'^\d+,\d+(,\d+)?$', re.IGNORECASE)

    with open(emoticons_data_file) as emoticons_file:
        emoticons_list = emoticons_file.read().splitlines()

    progress_bar = helper.ProgressBar(len(posts))
    for post in posts:
        tokens = post.tokens

        # remove urls
        # XXX: not sure if this removes important words! => word.startswith("https://")
        tokens = [word for word in tokens if regex_url.match(word) is None]

        # also remove www-links that do not start with "http://"!!
        tokens = filter(lambda t: not t.startswith("www."), tokens)

        # remove emoticons (list from https://en.wikipedia.org/wiki/List_of_emoticons)
        tokens = [word for word in tokens if word not in emoticons_list]

        # remove emoticons (regex)
        tokens = [word for word in tokens if regex_emoticons.match(word) is None]

        # remove words that start or end with "_"
        tokens = filter(lambda t: not t.startswith("_") and not t.endswith("_"), tokens)

        new_tokens = []
        for t in tokens:
            if t in tag_names:
                new_tokens.append([t])
                continue

            separator = None
            for sep in ["-", ".", "_", ",", "/"]:
                if sep in t:
                    separator = sep
                    break
            if separator is None:
                new_tokens.append([t])
                continue

            parts = t.split(separator)
            assert len(parts) > 1
            if len(parts) == 2 and separator == ".":
                if parts[1] in KNOWN_FILE_EXTENSIONS_MAP:
                    new_tokens.append(KNOWN_FILE_EXTENSIONS_MAP[parts[1]])
                    continue
#                 else:
#                     print parts[1]
            new_tokens.append(parts)
        tokens = [t for sub_tokens in new_tokens for t in sub_tokens]

        # remove empty tokens, numbers and those single-character words that are no letters
        tokens = filter(lambda t: len(t) > 0 and not t.isdigit() and len(single_character_tokens_re.findall(t)) == 0,
                        tokens)

        # remove single- and dual-character words that are not part of our tag list
        tokens = filter(lambda t: len(t) > 2 or t in tag_names, tokens)
#         # remove single-character words that are not part of our tag list
#         tokens = filter(lambda t: len(t) > 1 or t in tag_names, tokens)

        # remove "'s" at the end
        # "'s" is not useful for us because it may only indicate the verb "be", "have"
        # (e.g. "He's got ..." or "It's ...")
        # or may indicate a possessive nouns (e.g. His wife's shoes...)
        tokens = map(lambda t: t[:-2] if t.endswith("'s") else t, tokens)

        # remove "s'" at the end
        # "s'" is also not useful for us since it indicates the plural version
        # of possessive nouns (e.g. the planets' orbits)
        tokens = map(lambda t: t[:-2] if t.endswith("s'") else t, tokens)

        # remove "'ve", "'ed" "'re", "'ll" at the end (e.g. we've got...)
        tokens = map(lambda t: t[:-3] if t.endswith("'ve") or t.endswith("'ed") or t.endswith("'re") or t.endswith("'ll") else t, tokens)

        # remove "'d" and "'m" at the end (e.g. we'd...)
        tokens = map(lambda t: t[:-2] if t.endswith("'d") or t.endswith("'m") else t, tokens)

        # remove words ending with "n't" at the end (e.g. isn't)
        tokens = filter(lambda t: not t.endswith("n't"), tokens)

        # also remove numbers starting with #
        tokens = [word for word in tokens if regex_number.match(word) is None]

        # remove hexadecimal numbers
        tokens = [word for word in tokens if regex_hex_numbers.match(word) is None]

        # remove hexadecimal color_codes
        tokens = [word for word in tokens if regex_color_code.match(word) is None]

        # remove . and , separated numbers and enumerations!
        tokens = filter(lambda t: regex_float_number.match(t) is None, tokens)
        tokens = filter(lambda t: regex_long_number_in_separated_format.match(t) is None, tokens)

        # remove @-mentions
        tokens = filter(lambda t: not t.startswith("@"), tokens)

        # make sure that all tokens do not contain any whitespaces before and at the end
        tokens = map(lambda t: t.strip(), tokens)

        post.tokens = tokens
        progress_bar.update()

    progress_bar.finish()

    # TODO: remove very unique words that only occur once in the whole dataset _filter_tokens ??!!
    return


def filter_less_relevant_posts(posts, score_threshold):
    _logger.info("Filtering less relevant posts according to #answers and score value")
    # filter posts having low score according to given threshold
    posts = filter(lambda p: p.score >= score_threshold, posts)
    # posts = filter(lambda p: p.accepted_answer_id is not None, posts)
    posts = filter(lambda p: len(p.answers) > 0 or p.score > 0, posts)
    return posts
