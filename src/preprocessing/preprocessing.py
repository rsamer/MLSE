# -*- coding: utf-8 -*-

import logging
import os, csv
from entities.tag import Tag
from entities.post import Post
from util import helper
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


def replace_adjacent_token_synonyms_and_remove_adjacent_stopwords(posts):
    '''
        Looks for adjacent tokens in each post as defined in the synonym list
        and replaces the synonyms according to the synonym list.

        Note: Synonyms that are assigned to no/empty target word in the list are considered
              as 2/3-gram stopwords and removed.

        The synonym list mainly covers the most frequent 1-gram, 2-gram and 3-grams
        of the whole 'programmers.stackexchange.com' dataset (after our tokenization,
        stopword-removal, ...) as analyzed by using scikitlearn's Count-vectorizer.

        --------------------------------------------------------------------------------------------
        NOTE: Please keep in mind that this method is executed BEFORE stemming, so the list
              may contain slightly different versions of the same synonym words (e.g. plurals, ...)
              This is useful for some context-based words where stemming fails.

              Doing the synonym replacement step before stemming makes the synonym list much more
              readable.
        --------------------------------------------------------------------------------------------
    '''
    synonyms_file_path = os.path.join(helper.APP_PATH, 'corpora', 'tokens', 'synonyms')
    token_replacement_map_unigram = {}
    token_replacement_map_bigram = {}
    token_replacement_map_trigram = {}
    with open(synonyms_file_path, 'rb') as f:
        reader = csv.reader(f)
        for row in reader:
            source_token_ngram = row[1].strip()
            source_token_parts = source_token_ngram.split()
            target_token_parts = row[0].strip().split()
            if len(source_token_parts) == 1:
                assert row[1] not in token_replacement_map_unigram, "Synonym entry '%s' is ambiguous." % row[1]
                token_replacement_map_unigram[source_token_ngram] = target_token_parts
            elif len(source_token_parts) == 2:
                assert row[1] not in token_replacement_map_bigram, "Synonym entry '%s' is ambiguous." % row[1]
                token_replacement_map_bigram[source_token_ngram] = target_token_parts
            elif len(source_token_parts) == 3:
                assert row[1] not in token_replacement_map_trigram, "Synonym entry '%s' is ambiguous." % row[1]
                token_replacement_map_trigram[source_token_ngram] = target_token_parts
            else:
                assert False, "Invalid entry in synonyms list! Only supported: unigrams, bigrams, trigrams"

    n_replacements_total = 0
    for post in posts:
        assert isinstance(post, Post)

        def _replace_token_list_synonyms(tokens, token_replacement_map, n_gram=1):
            assert isinstance(tokens, list)
            n_replacements = 0
            if len(tokens) < n_gram:
                return (tokens, n_replacements)

            new_tokens = []
            skip_n_tokens = 0
            for i in range(len(tokens)):
                # simplify in order to avoid redundant loop iterations...
                if skip_n_tokens > 0:
                    skip_n_tokens -= 1
                    continue

                if i + n_gram > len(tokens):
                    new_tokens += tokens[i:]
                    break

                n_gram_word = ' '.join(tokens[i:i+n_gram])
                if n_gram_word in token_replacement_map:
                    new_tokens += token_replacement_map[n_gram_word]
                    skip_n_tokens = (n_gram - 1)
                    n_replacements += 1
                else:
                    new_tokens += [tokens[i]]
            return (new_tokens, n_replacements)

        # title tokens
        tokens = post.title_tokens
        tokens, n_replacements_trigram = _replace_token_list_synonyms(tokens, token_replacement_map_trigram, n_gram=3)
        assert isinstance(tokens, list)
        tokens, n_replacements_bigram = _replace_token_list_synonyms(tokens, token_replacement_map_bigram, n_gram=2)
        tokens, n_replacements_unigram = _replace_token_list_synonyms(tokens, token_replacement_map_unigram, n_gram=1)
        # adjacent stop words have been replaced with empty string! -> remove empty tokens now!
        tokens = filter(lambda t: len(t) > 0, tokens)
        post.title_tokens = tokens
        n_replacements_total += n_replacements_trigram + n_replacements_bigram + n_replacements_unigram

        # body tokens
        tokens = post.body_tokens
        tokens, n_replacements_trigram = _replace_token_list_synonyms(tokens, token_replacement_map_trigram, n_gram=3)
        tokens, n_replacements_bigram = _replace_token_list_synonyms(tokens, token_replacement_map_bigram, n_gram=2)
        tokens, n_replacements_unigram = _replace_token_list_synonyms(tokens, token_replacement_map_unigram, n_gram=1)
        # adjacent stop words have been replaced with empty string! -> remove empty tokens now!
        tokens = filter(lambda t: len(t) > 0, tokens)
        post.body_tokens = tokens
        n_replacements_total += n_replacements_trigram + n_replacements_bigram + n_replacements_unigram

    _logger.info("Found and replaced %s synonym tokens", n_replacements_total)

def important_words_for_tokenization(tag_names):
    '''
        Collects important words that must NOT be destroyed by tokenization and filtering process.
        Note: Since the synonym list also contains adjacent-stopwords (those that have
              no target word in the list), only synonym-words (target and source) are included.

              Synonyms are then replaced after tokenization.
              (see: replace_adjacent_token_synonyms_and_remove_adjacent_stopwords()-method above)
    '''
    custom_important_words = []
    synonyms_file_path = os.path.join(helper.APP_PATH, 'corpora', 'tokens', 'synonyms')
    with open(synonyms_file_path, 'rb') as f:
        reader = csv.reader(f)
        for row in reader:
            target_word, source_word = row[0].strip().lower(), row[1].strip().lower()
            if len(target_word) > 0: # only add synonym words and not adjacent stopwords
                custom_important_words += filter(lambda w: not helper.is_int_or_float(w),
                                                 target_word.split() + source_word.split())

    tag_names = map(lambda n: n.lower(), tag_names)
    return list(set(tag_names + map(lambda w: w.lower(), custom_important_words)))


def preprocess_posts(posts, tag_list, filter_posts=True, enable_stemming=True,
                     replace_adjacent_tag_occurences=True,
                     replace_token_synonyms_and_remove_adjacent_stopwords=True):
    _logger.info("Preprocessing posts")
    assert isinstance(posts, list)

    if filter_posts is True:
        posts = filters.filter_less_relevant_posts(posts, 0)
        posts = tags.strip_invalid_tags_from_posts_and_remove_untagged_posts(posts, tag_list)

    assert len(posts) > 0, "No posts given. All posts have been filtered out. Please check your parameters!"

    # TODO: @Michael: why is this module named selection??!
    selection.append_accepted_answer_text_to_body(posts)

    filters.to_lower_case(posts)
    filters.strip_code_segments(posts)
    filters.strip_html_tags(posts)

    tag_names = map(lambda t: t.name.lower(), tag_list)
    if replace_adjacent_tag_occurences:
        tags.replace_adjacent_tag_occurences(posts, tag_names)

    important_words = important_words_for_tokenization(tag_names)
    _logger.info("Number of important words {} (altogether)".format(len(important_words)))
    tokenizer.tokenize_posts(posts, important_words)
    n_tokens = reduce(lambda x,y: x + y, map(lambda t: len(t.title_tokens) + len(t.body_tokens), posts))
    filters.filter_tokens(posts, tag_names)

    stopwords.remove_stopwords(posts, tag_names)

    if replace_token_synonyms_and_remove_adjacent_stopwords is True:
        replace_adjacent_token_synonyms_and_remove_adjacent_stopwords(posts)

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
