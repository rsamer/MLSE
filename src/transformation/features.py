# -*- coding: utf-8 -*-

import logging
from entities.post import Post
from entities.tag import Tag
from time import time
import numpy as np

_logger = logging.getLogger(__name__)

def numeric_features(train_posts, test_posts, tag_list):
    _logger.info("Numeric features (Transformation)")

    assert(isinstance(train_posts, list))
    assert(isinstance(test_posts, list))
    assert(isinstance(tag_list, list))

    def extract_features(post_list, tag_list):
        X = []

        for post in post_list:
            feature_list = []

            # feature 1: does title contain tag x
            for tag in tag_list:
                feature_list += int(tag.name in post.title)

            # feature 2: does body contain tag x
            for tag in tag_list:
                feature_list += int(tag.name in post.tokens)

            X += [np.array(feature_list)]

        return X

    X_test = extract_features(test_posts, tag_list)
    X_train = extract_features(train_posts, tag_list)

    return X_train, X_test
