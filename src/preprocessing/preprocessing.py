# -*- coding: utf-8 -*-

import logging
from entities.tag import Tag
import tokenizer
import filters
import tags
import stopwords
import stemmer
import selection
import pos
import lemmatizer
import itertools

_logger = logging.getLogger(__name__)

def filter_tags_and_sort_by_frequency(tags, frequency_threshold):
    '''
        Sorts tags by frequency and removes less frequent tags according to given threshold.
        Finally, unassigns removed tags from given post-list.
    '''
    _logger.info("Filter tags and sort by frequency - Frequency-Threshold = %d", frequency_threshold)
    assert isinstance(tags, list)
    reverse_sorted_tags = Tag.sort_tags_by_frequency(tags, reverse=True) # descendent order

    if frequency_threshold <= 1:
        return reverse_sorted_tags

    return list(itertools.takewhile(lambda t: t.count >= frequency_threshold, iter(reverse_sorted_tags)))


def stem_tags(tags):
    stemmer.porter_stemmer_tags(tags)


def preprocess_posts(posts, tag_list, filter_posts=True, enable_stemming=True,
                     replace_adjacent_tag_occurences=True):
    _logger.info("Preprocessing posts")
    assert isinstance(posts, list)

    if filter_posts is True:
        posts = filters.filter_less_relevant_posts(posts, 0)
        posts = tags.strip_invalid_tags_from_posts_and_remove_untagged_posts(posts, tag_list)

    assert len(posts) > 0, "No posts given. All posts have been filtered out. Please check your parameters!"

    # TODO: why is this module named selection??!
    selection.append_accepted_answer_text_to_body(posts)

    filters.to_lower_case(posts)
    filters.strip_code_segments(posts)
    filters.strip_html_tags(posts)

    tag_names = map(lambda t: t.name.lower(), tag_list)
    if replace_adjacent_tag_occurences:
        tags.replace_adjacent_tag_occurences(posts, tag_names)

    tokenizer.tokenize_posts(posts, tag_names)
    n_tokens = reduce(lambda x,y: x + y, map(lambda t: len(t.title_tokens) + len(t.body_tokens), posts))
    filters.filter_tokens(posts, tag_names)

    stopwords.remove_stopwords(posts, tag_names)
    #pos.pos_tagging(posts)

    #-----------------------------------------------------------------------------------------------
    # NOTE: it does not makes sense to use both lemmatization and stemming
    #       lemmatizer also requires pos_tagging beforehand!
    # lemmatizer.word_net_lemmatizer(posts)
    #-----------------------------------------------------------------------------------------------

    if enable_stemming is True:
        stemmer.porter_stemmer(posts)

    n_filtered_tokens = n_tokens - reduce(lambda x,y: x + y, map(lambda t: len(t.title_tokens) + len(t.body_tokens), posts))
    if n_tokens > 0:
        _logger.info("Removed {} ({}%) of {} tokens (altogether)".format(n_filtered_tokens,
                        round(float(n_filtered_tokens) / n_tokens * 100.0, 2), n_tokens))
    return posts
