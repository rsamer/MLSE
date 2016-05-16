# -*- coding: utf-8 -*-

import numpy


def precision(posts):
    """
    Computes the precision for a tag prediction.
    Make sure these attributes of the post class are set: tag_set, tag_set_predicted
    """
    assert isinstance(posts, list)

    # https://pdfs.semanticscholar.org/b7c5/3b62e037180e42b59e5cbb5ed953c6bb00e6.pdf
    # Relative precision is calculated for each post comparing the predicted tags with the "hidden" real tags
    # Overall precision is the mean precision of all posts.

    precisions = []

    for post in posts:
        true_positive = 0
        false_positive = 0
        
        for tag in post.tag_set_prediction:
            if post.contains_tag_with_name(tag.name):
                true_positive += 1
            else:
                false_positive += 1

        p = float(true_positive) / float(true_positive + false_positive) if len(post.tag_set_prediction) > 0 else 0
        precisions.append(p)

    overall_precision = numpy.mean(precisions)
    return overall_precision
