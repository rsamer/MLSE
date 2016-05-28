# -*- coding: utf-8 -*-

import re
import logging
from entities.tag import Tag
import tokenizer
import filters
import tags
import stopwords
import stemmer
import pos
import lemmatizer
import nltk
import os
import itertools

_logger = logging.getLogger(__name__)
main_dir = os.path.dirname(os.path.realpath(__file__)) + "/../../"
nltk.data.path = [main_dir + "corpora/nltk_data"]
emoticons_data_file = main_dir + "corpora/emoticons/emoticons"
tokens_punctuation_re = re.compile(r"(\.|!|\?|\(|\)|~)$")
single_character_tokens_re = re.compile(r"^\W$")


def filter_tags_and_sort_by_frequency(tags, frequency_threshold):
    '''
        Sorts tags by frequency and removes less frequent tags according to given threshold.
        Finally, unassigns removed tags from given post-list.
    '''
    _logger.info("-"*80)
    _logger.info("Filter tags and sort by frequency")
    _logger.info("-"*80)
    assert isinstance(tags, list)
    reverse_sorted_tags = Tag.sort_tags_by_frequency(tags, reverse=True) # descendent order

    if frequency_threshold <= 1:
        return reverse_sorted_tags

    # TODO: (end-term) look for similar tag names and merge them together, e.g. "java", "java programming"??
    #       for this we should simply download the synonym list from StackExchange as mentioned in the paper!
    return list(itertools.takewhile(lambda tag: tag.count >= frequency_threshold, iter(reverse_sorted_tags)))


def preprocess_posts(posts, tag_list, filter_untagged_posts=True, filter_less_relevant_posts=True):

    _logger.info("-"*80)
    _logger.info("Preprocessing posts")
    _logger.info("-"*80)

    assert isinstance(posts, list)
    tag_names = [tag.name.lower() for tag in tag_list]

    filters.add_accepted_answer_text_to_body(posts) # XXX: this is no filtering function...??

    if filter_less_relevant_posts:
        posts = filters.filter_less_relevant_posts(posts, 0)
    if filter_untagged_posts:
        posts = tags.strip_invalid_tags_from_posts_and_remove_untagged_posts(posts, tag_list)

    filters.to_lower_case(posts)
    filters.strip_code_segments(posts)
    filters.strip_html_tags(posts)
    tags.replace_adjacent_tag_occurences(posts, tag_names)
    tokenizer.tokenize_posts(posts, tag_names)
    n_tokens = reduce(lambda x,y: x + y, map(lambda t: len(t.tokens), posts))
    filters.filter_tokens(posts, tag_names)
    stopwords.remove_stopwords(posts)
    pos.pos_tagging(posts)
    n_filtered_tokens = n_tokens - reduce(lambda x,y: x + y, map(lambda t: len(t.tokens), posts))
    if n_tokens > 0:
        _logger.info("Removed {} ({}%) of {} tokens (altogether)".format(n_filtered_tokens,
                        round(float(n_filtered_tokens) / n_tokens * 100.0, 2), n_tokens))
    #lemmatizer.word_net_lemmatizer(posts) # it does not makes sense to use both lemmatization and stemming
    stemmer.porter_stemmer(posts)
    return posts
