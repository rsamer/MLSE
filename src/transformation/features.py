# -*- coding: utf-8 -*-

import logging
from entities.post import Post
from entities.tag import Tag
from time import time
import numpy as np
from scipy import sparse
from math import log
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

    for token in title_token_occurrence:
        p_title_token[token] = title_token_occurrence[token] / len(title_token_occurrence)

    for token in body_token_occurrence:
        p_body_token[token] = body_token_occurrence[token] / len(body_token_occurrence)

    for tag_name in tag_occurrence:
        p_tag[tag_name] = tag_occurrence[tag_name] / len(tag_occurrence)

    for tag_token in tag_title_token_occurrence:
        p_tag_title_token[tag_token] = tag_title_token_occurrence[tag_token] / len(tag_title_token_occurrence)

    for tag_token in tag_body_token_occurrence:
        p_tag_body_token[tag_token] = tag_body_token_occurrence[tag_token] / len(tag_body_token_occurrence)

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
                    # feature 3+4: does title or body contain relaxed tag x
                    relaxed_words = tag_name.split("-")
                    n_occurrences_title = len(filter(lambda w: w in post.title_tokens, relaxed_words))
                    n_occurrences_body = len(filter(lambda w: w in post.body_tokens, relaxed_words))
                    feature_list += [int(n_occurrences_title == len(relaxed_words))]
                    feature_list += [int(n_occurrences_body == len(relaxed_words))]

                # feature 5: title PMI
                title_pmi = 0.0
                for token in post.title_tokens:
                    tag_token = "_".join([tag_name, token])
                    if tag_token in p_tag_title_token:
                        title_pmi += log(p_tag_title_token[tag_token] / (p_tag[tag_name] * p_title_token[token]))
                feature_list += [title_pmi]

                # feature 6: body PMI
                body_pmi = 0.0
                for token in post.body_tokens:
                    tag_token = "_".join([tag_name, token])
                    if tag_token in p_tag_body_token:
                        body_pmi += log(p_tag_body_token[tag_token] / (p_tag[tag_name] * p_body_token[token]))
                feature_list += [body_pmi]

            X += [np.array(feature_list)]

        # TODO: X = StandardScaler().fit_transform(X)
        return sparse.csr_matrix(X)

    X_train = extract_features(train_posts)
    X_test = extract_features(test_posts)

    return X_train, X_test
