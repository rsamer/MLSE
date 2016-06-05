# -*- coding: utf-8 -*-

import logging
from entities.post import Post
from entities.tag import Tag
from time import time
import numpy as np
from sklearn.preprocessing import StandardScaler

_logger = logging.getLogger(__name__)

def numeric_features(train_posts, test_posts, tag_list):
    _logger.info("Numeric features (Transformation)")
    assert isinstance(train_posts, list)
    assert isinstance(test_posts, list)
    assert isinstance(tag_list, list)

    def extract_features(post_list, tag_list):
        X = []

        for post in post_list:
            feature_list = []
            for tag in tag_list:
                # feature 1: does title contain tag x
                feature_list += [int(tag.name in post.title_tokens)]
                # feature 2: does body contain tag x
                feature_list += [int(tag.name in post.body_tokens)]

                if "-" in tag.name:
                    # feature 3: does title or body contain relaxed tag x
                    relaxed_words = tag.name.split("-")
                    n_occurences_title = len(filter(lambda w: w in post.title_tokens, relaxed_words))
                    n_occurences_body = len(filter(lambda w: w in post.body_tokens, relaxed_words))
                    feature_list += [n_occurences_title == len(relaxed_words)]
                    feature_list += [n_occurences_body == len(relaxed_words)]
            X += [np.array(feature_list)]
        # TODO: X = StandardScaler().fit_transform(X)
        return X

    X_test = extract_features(test_posts, tag_list)
    X_train = extract_features(train_posts, tag_list)

    return X_train, X_test
