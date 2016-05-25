# -*- coding: utf-8 -*-

import logging
import nltk
import os

main_dir = os.path.dirname(os.path.realpath(__file__)) + "/../../"
nltk.data.path = [main_dir + "corpora/nltk_data"]

log = logging.getLogger("preprocessing.pos")


# TODO: Stanford POS-tagging
def pos_tagging(posts):
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
    print "=" * 80 + "\n\n"
    print existing_pos_tags
    print removed_tokens