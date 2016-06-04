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
    _logger.info("Filter tags and sort by frequency")
    assert isinstance(tags, list)
    reverse_sorted_tags = Tag.sort_tags_by_frequency(tags, reverse=True) # descendent order

    if frequency_threshold <= 1:
        return reverse_sorted_tags

    # TODO: (end-term) look for similar tag names and merge them together, e.g. "java", "java programming"??
    #       for this we should simply download the synonym list from StackExchange as mentioned in the paper!
    return list(itertools.takewhile(lambda tag: tag.count >= frequency_threshold, iter(reverse_sorted_tags)))


def preprocess_tags(tags):
    stemmer.porter_stemmer_tags(tags)


def preprocess_posts(posts, tag_list, filter_posts=True):
    _logger.info("Preprocessing posts")
    assert isinstance(posts, list)

    if filter_posts:
        posts = filters.filter_less_relevant_posts(posts, 0)
        posts = tags.strip_invalid_tags_from_posts_and_remove_untagged_posts(posts, tag_list)

    selection.add_title_to_body(posts, 3)
    selection.add_accepted_answer_text_to_body(posts)

    filters.to_lower_case(posts)
    filters.strip_code_segments(posts)
    filters.strip_html_tags(posts)

    tag_names = map(lambda t: t.name.lower(), tag_list)
    tags.replace_adjacent_tag_occurences(posts, tag_names)

    tokenizer.tokenize_posts(posts, tag_names)
    n_tokens = reduce(lambda x,y: x + y, map(lambda t: len(t.tokens), posts))
    filters.filter_tokens(posts, tag_names)

    stopwords.remove_stopwords(posts)
    #pos.pos_tagging(posts)

    n_filtered_tokens = n_tokens - reduce(lambda x,y: x + y, map(lambda t: len(t.tokens), posts))
    if n_tokens > 0:
        _logger.info("Removed {} ({}%) of {} tokens (altogether)".format(n_filtered_tokens,
                        round(float(n_filtered_tokens) / n_tokens * 100.0, 2), n_tokens))
    #lemmatizer.word_net_lemmatizer(posts) # it does not makes sense to use both lemmatization and stemming
    stemmer.porter_stemmer(posts)
    return posts
