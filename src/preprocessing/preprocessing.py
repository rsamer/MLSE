# -*- coding: utf-8 -*-

import re
import logging
from entities.tag import Tag
from entities.post import Post
import nltk
import os
import itertools

main_dir = os.path.dirname(os.path.realpath(__file__)) + "/../../"
nltk.data.path = [main_dir + "corpora/nltk_data"]
emoticons_data_file = main_dir + "corpora/emoticons/emoticons"

log = logging.getLogger("preprocessing.preprocessing")

tokens_punctuation_re = re.compile(r"(\.|!|\?|\(|\)|~)$")
single_character_tokens_re = re.compile(r"^\W$")


def filter_tags_and_sort_by_frequency(tags, frequency_threshold):
    ''' Sorts tags by frequency and removes less frequent tags according to given threshold.
        Finally, unassigns removed tags from given post-list.
    '''
    logging.info("-"*80)
    logging.info("Filter tags and sort by frequency")
    logging.info("-"*80)
    assert isinstance(tags, list)
    reverse_sorted_tags = Tag.sort_tags_by_frequency(tags, reverse=True) # descendent order

    if frequency_threshold <= 1:
        return reverse_sorted_tags

    # TODO: (end-term) look for similar tag names and merge them together, e.g. "java", "java programming"??
    #       for this we should simply download the synonym list from StackExchange as mentioned in the paper!
    return list(itertools.takewhile(lambda tag: tag.count >= frequency_threshold, iter(reverse_sorted_tags)))


def preprocess_posts(posts, tags, filter_untagged_posts=True, filter_less_relevant_posts=True):

    logging.info("-"*80)
    logging.info("Preprocessing posts")
    logging.info("-"*80)

    def _add_accepted_answer_text_to_body(posts):
        for post in posts:
            if post.accepted_answer_id is None:
                continue
            accepted_answer = post.accepted_answer()
            if accepted_answer is None:
                continue
            #assert accepted_answer is not None
            if accepted_answer.score >= 0:
                post.body += " " + accepted_answer.body

    def _strip_invalid_tags_from_posts_and_remove_untagged_posts(posts, tags):
        ''' unassigns all removed tags from posts to avoid data-inconsistency issues '''
        logging.info("Stripping invalid tags from posts and removing untagged posts")
        new_post_list = []
        for post in posts:
            post.tag_set = post.tag_set.intersection(tags)
            if len(post.tag_set) > 0:
                new_post_list.append(post)
        return new_post_list


    def _filter_less_relevant_posts(posts, score_threshold):
        logging.info("Filtering less relevant posts according to #answers and score value")
        # filter posts having low score according to given threshold
        posts = filter(lambda p: p.score >= score_threshold, posts)
        posts = filter(lambda p: p.accepted_answer_id is not None, posts)
        posts = filter(lambda p: len(p.answers) > 0 or p.score > 0, posts)
        return posts


    def _to_lower_case(posts):
        logging.info("Lower case post title and body")
        for post in posts:
            post.title = post.title.lower()
            post.body = post.body.lower()


    def _strip_code_segments(posts):
        logging.info("Stripping code snippet from posts")
        for post in posts:
            assert (isinstance(post, Post))
            post.body = re.sub('<code>.*?</code>', '', post.body)


    def _strip_html_tags(posts):
        logging.info("Stripping HTML-tags from posts")
        try:
            from bs4 import BeautifulSoup #@UnresolvedImport
        except ImportError:
            raise RuntimeError('Please install BeautifulSoup library!')
    
        for post in posts:
            assert(isinstance(post, Post))
            post.title = BeautifulSoup(post.title, "html.parser").text.strip()
            post.body = BeautifulSoup(post.body, "html.parser").text.strip()


    def _replace_adjacent_tag_occurences(posts, tag_names):
        ''' replaces "-" by " " in all tag names e.g. "object-oriented" -> "object oriented"
            and then looks for two (or more) adjacent words that represent a known tag name
            e.g. current token list ["I", "love", "object", "oriented", "code"]
            -> should be converted to ["I", "love", "object-oriented", "code"]
            since "object-oriented" is a tag name in our tag list
        '''
        for tag_name in tag_names:
            splitted_tag_name = tag_name.replace("-", " ")
            if splitted_tag_name == tag_name:
                continue
            for post in posts:
                post.body = post.body.replace(splitted_tag_name, tag_name)
                

    def _tokenize_posts(posts, tag_names):
        ''' Customized tokenizer for our special needs! (e.g. C#, C++, ...) '''
        logging.info("Tokenizing posts")
        # based on: http://stackoverflow.com/a/36463112
        regex_str = [
            #r'(?:[:;=\^\-oO][\-_\.]?[\)\(\]\[\-DPOp_\^\\\/])', # emoticons (Note: they are removed after tokenization!) # TODO why not here?
            r'\w+\.\w+',
            r'\w+\&\w+',    # e.g. AT&T
            r'(?:@[\w_]+)', # @-mentions
            r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", # hash-tags
            r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', # URLs  (Note: they are removed after tokenization!)
            #r'(?:\d+\%)', # percentage
            r'(?:(?:\d+,?)+(?:\.?\d+)?)', # numbers
            r"(?:[a-z][a-z'\-_]+[a-z])", # words with - and '
        ] + map(lambda tag_name: re.escape(tag_name) + '\W', tag_names) + [ # consider known tag names
            r'(?:[\w_]+)', # other words
            r'(?:\S)' # anything else
        ]
        tokens_ignore_re = re.compile(r'('+'|'.join(regex_str)+')', re.VERBOSE | re.IGNORECASE)

        def _tokenize_text(s, tag_names):
            def tokenize(s):
                return tokens_ignore_re.findall(s)
    
            def filter_all_tailing_punctuation_characters(token):
                while True:
                    punctuation_character = tokens_punctuation_re.findall(token)
                    if len(punctuation_character) == 0:
                        return token
                    token = token[:-1]

            # prepending and appending a single whitespace to the text
            # makes the regular expressions less complicated
            tokens = tokenize(" " + s + " ")
            tokens = [token.strip() for token in tokens] # remove whitespaces before and after
            tokens = map(filter_all_tailing_punctuation_characters, tokens)
            return tokens

        for post in posts:
            text = ((post.title + " ") * 10) + post.body
            post.tokens = _tokenize_text(text, tag_names)


    def _filter_tokens(posts, tag_names):
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
            tokens = filter(lambda t: len(t) > 0 and not t.isdigit() and len(single_character_tokens_re.findall(t)) == 0, tokens)
    
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
            #tokens = [word for word in tokens if word not in emoticons_list]

            # remove emoticons (regex)
            #tokens = [word for word in tokens if regex_emoticons.match(word) is None]

            post.tokens = tokens
            total_number_of_filtered_tokens += (num_of_unfiltered_tokens - len(post.tokens))

        if total_number_of_tokens != 0:
            logging.info("Removed {} ({}%) of {} tokens (altogether)".format(total_number_of_filtered_tokens,
                         round(float(total_number_of_filtered_tokens)/total_number_of_tokens*100.0, 2),
                         total_number_of_tokens))


    def _remove_stopwords(posts):
        logging.info("Removing stop-words from posts' tokens")
        try:
            from nltk.corpus import stopwords
        except ImportError:
            raise RuntimeError('Please install nltk library!')

        stop_words = stopwords.words('english')

        for post in posts:
            post.tokens = [word for word in post.tokens if word not in stop_words]


    def _stemming(posts):
        logging.info("Stemming for posts' tokens")
        try:
            from nltk.stem.porter import PorterStemmer
        except ImportError:
            raise RuntimeError('Please install nltk library!')

        porter = PorterStemmer()

        for post in posts:
            post.tokens = [porter.stem(word) for word in post.tokens]


    def _lemmatization(posts):
        logging.info("Lemmatization for posts' tokens")
        try:
            from nltk.stem.wordnet import WordNetLemmatizer
        except ImportError:
            raise RuntimeError('Please install nltk library!')

        lemmatizer = WordNetLemmatizer()

        for post in posts:
            post.tokens = [lemmatizer.lemmatize(word) for word in post.tokens]

    def _pos_tagging(posts):
        logging.info("Pos-tagging for posts' tokens")
        try:
            import nltk
        except ImportError:
            raise RuntimeError('Please install nltk library!')

        '''
        Tag    Meaning    English Examples
        ADJ    adjective    new, good, high, special, big, local
        ADP    adposition    on, of, at, with, by, into, under
        ADV    adverb    really, already, still, early, now
        CONJ    conjunction    and, or, but, if, while, although
        DET    determiner, article    the, a, some, most, every, no, which
        NOUN    noun    year, home, costs, time, Africa
        NUM    numeral    twenty-four, fourth, 1991, 14:24
        PRT    particle    at, on, out, over per, that, up, with
        PRON    pronoun    he, their, her, its, my, I, us
        VERB    verb    is, say, told, given, playing, would
        .    punctuation marks    . , ; !
        X    other    ersatz, esprit, dunno, gr8, univeristy
        '''
        pos_tags_black_list = ['CONJ', 'DET', 'PRT', 'PRON', '.']
        existing_pos_tags = set()
        removed_tokens = set()
        for post in posts:
            pos_tagged_tokens = nltk.pos_tag(post.tokens, tagset='universal')
            tagged_tokens = filter(lambda t: t[1] not in pos_tags_black_list, pos_tagged_tokens)
            post.tokens = map(lambda t: t[0], tagged_tokens)
            post.tokens_pos_tags = map(lambda t: t[1], tagged_tokens)
            removed_tokens |= set(filter(lambda t: t[1] in pos_tags_black_list, pos_tagged_tokens))
            existing_pos_tags |= set(map(lambda t: t[1], pos_tagged_tokens))
        print "="*80 + "\n\n"
        print existing_pos_tags
        print removed_tokens

    assert isinstance(posts, list)
    tag_names = [tag.name.lower() for tag in tags]
    _add_accepted_answer_text_to_body(posts)
    if filter_less_relevant_posts:
        posts = _filter_less_relevant_posts(posts, 0)
    if filter_untagged_posts:
        posts = _strip_invalid_tags_from_posts_and_remove_untagged_posts(posts, tags)
    _to_lower_case(posts)
    _strip_code_segments(posts)
    _strip_html_tags(posts)
    _replace_adjacent_tag_occurences(posts, tag_names)
    _tokenize_posts(posts, tag_names)
    _filter_tokens(posts, tag_names)


    # TODO: remove emoticons in _filter_tokens (right AFTER tokenization and NOT before!!)
    # TODO: remove very unique words that only occur once in the whole dataset _filter_tokens ??!!

    _remove_stopwords(posts)
    #_pos_tagging(posts)
    #_lemmatization(posts) # not sure if it makes sense to use both lemmatization and stemming
    _stemming(posts)
    return posts

