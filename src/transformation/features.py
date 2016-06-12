# -*- coding: utf-8 -*-

import logging
import numpy as np
from scipy import sparse
import math
from math import log, exp
from sklearn.preprocessing import StandardScaler

_logger = logging.getLogger(__name__)


def numeric_features(train_posts, test_posts, tag_list):
    _logger.info("Numeric features (Transformation)")
    assert isinstance(train_posts, list)
    assert isinstance(test_posts, list)
    assert isinstance(tag_list, list)

    tag_occurrence = {}
    p_tag = {}
    title_token_occurrence = {}
    tag_title_token_occurrence = {}
    p_title_token = {}
    p_tag_title_token = {}
    body_token_occurrence = {}
    tag_body_token_occurrence = {}
    p_body_token = {}
    p_tag_body_token = {}

    # compute occurences
    for post in train_posts:
        for tag in post.tag_set:
            tag_name = tag.preprocessed_tag_name
            if tag_name not in tag_occurrence:
                tag_occurrence[tag_name] = 0.0
            tag_occurrence[tag_name] += 1.0

            for token in post.title_tokens:
                tag_token = "_".join([tag_name, token])
                if tag_token not in tag_title_token_occurrence:
                    tag_title_token_occurrence[tag_token] = 0.0
                tag_title_token_occurrence[tag_token] += 1.0

            for token in post.body_tokens:
                tag_token = "_".join([tag_name, token])
                if tag_token not in tag_body_token_occurrence:
                    tag_body_token_occurrence[tag_token] = 0.0
                tag_body_token_occurrence[tag_token] += 1.0

        for token in post.title_tokens:
            if token not in title_token_occurrence:
                title_token_occurrence[token] = 0.0
            title_token_occurrence[token] += 1.0

        for token in post.body_tokens:
            if token not in body_token_occurrence:
                body_token_occurrence[token] = 0.0
            body_token_occurrence[token] += 1.0


    # compute probabilities
    n_total_title_token_occurences = reduce(lambda x,y: x+y, title_token_occurrence.values())
    # fixed bug! -> n_total_title_token_occurences = len(title_token_occurrence)
    for token, occurrence in title_token_occurrence.iteritems():
        p_title_token[token] = occurrence / n_total_title_token_occurences
        assert p_title_token[token] <= 1.0

    n_total_body_token_occurences = reduce(lambda x,y: x+y, body_token_occurrence.values())
    # fixed bug! -> n_total_body_token_occurences = len(body_token_occurrence)
    for token, occurrence in body_token_occurrence.iteritems():
        p_body_token[token] = occurrence / n_total_body_token_occurences
        assert p_body_token[token] <= 1.0

    n_total_tag_occurences = reduce(lambda x,y: x+y, tag_occurrence.values())
    # fixed bug! -> n_total_tag_occurences = len(tag_occurrence)
    for tag_name, occurrence in tag_occurrence.iteritems():
        p_tag[tag_name] = occurrence / n_total_tag_occurences
        assert p_tag[tag_name] <= 1.0

    n_total_tag_title_token_combinations = reduce(lambda x,y: x+y, tag_title_token_occurrence.values())
    # fixed bug! -> n_total_tag_title_token_combinations = len(tag_title_token_occurrence)
    for tag_token, occurrence in tag_title_token_occurrence.iteritems():
        p_tag_title_token[tag_token] = occurrence / n_total_tag_title_token_combinations
        assert p_tag_title_token[tag_token] <= 1.0

    n_total_tag_body_token_combinations = reduce(lambda x,y: x+y, tag_body_token_occurrence.values())
    # fixed bug! -> n_total_tag_body_token_combinations = len(tag_body_token_occurrence)
    for tag_token, occurrence in tag_body_token_occurrence.iteritems():
        p_tag_body_token[tag_token] = occurrence / n_total_tag_body_token_combinations
        assert p_tag_body_token[tag_token] <= 1.0

    #assert len(p_tag) == len(tag_list)

    def extract_features(post_list):
        X = []
        for post in post_list:
            feature_list = []
            for tag in tag_list:
                tag_name = tag.preprocessed_tag_name
                # feature 1: does title contain tag x
                feature_list += [int(tag_name in post.title_tokens)]
                # feature 2: does body contain tag x
                feature_list += [int(tag_name in post.body_tokens)]

                if "-" in tag_name:
                    relaxed_tag = tag_name.split("-")
                    assert len(relaxed_tag) > 0
                    occurrences_title = filter(lambda w: w in post.title_tokens, relaxed_tag)
                    occurrences_body = filter(lambda w: w in post.body_tokens, relaxed_tag)
                    # feature 3: does title contain relaxed tag x
                    feature_list += [int(set(occurrences_title) == set(relaxed_tag))]
                    # feature 4: does body contain relaxed tag x
                    feature_list += [int(set(occurrences_body) == set(relaxed_tag))]

                # feature 5: title PMI
                title_pmi = 0.0
                for token in post.title_tokens:
                    tag_token = "_".join([tag_name, token])
                    if tag_token in p_tag_title_token:
                        title_pmi += log(p_tag_title_token[tag_token] / (p_tag[tag_name] * p_title_token[token]), math.e)
                print "Title EXP: " + str(exp(title_pmi))
                assert exp(title_pmi) <= 1.0
                feature_list += [title_pmi]

                # feature 6: body PMI
                body_pmi = 0.0
                for token in post.body_tokens:
                    tag_token = "_".join([tag_name, token])
                    if tag_token in p_tag_body_token:
                        body_pmi += log(p_tag_body_token[tag_token] / (p_tag[tag_name] * p_body_token[token]), math.e)
                print "Body EXP: " + str(exp(body_pmi))
                assert exp(body_pmi) <= 1.0
                feature_list += [body_pmi]

            X += [np.array(feature_list)]

        # TODO: X = StandardScaler().fit_transform(X)
        return sparse.csr_matrix(X)

    X_train = extract_features(train_posts)
    X_test = extract_features(test_posts)
    return X_train, X_test
