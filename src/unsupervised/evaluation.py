# -*- coding: utf-8 -*-

import numpy


def precision(posts):
    """
    Computes the precision for a tag prediction.
    Make sure these attributes of the post class are set: tag_set, tag_set_predicted
    """
    assert isinstance(posts, list)

#     y_predictions = []
#     y_all = []
#     for post in posts:
#         for tag in post.tag_set:
#             y_all += [tag.name]
#             predicted_tag_names = map(lambda t: t.name, post.tag_set_prediction)
#             y_predictions += [tag.name if tag.name in predicted_tag_names else ""]
#     from sklearn import metrics
#     print y_all
#     print y_predictions
#     print ">>>>>>>>> " + str(metrics.precision_score(y_all, y_predictions, average='macro'))
#     print ">>>>>>>>> " + str(metrics.recall_score(y_all, y_predictions, average='macro'))
#    metrics.f1_score()


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


def recall(posts):
    """
    Computes the precision for a tag prediction.
    Make sure these attributes of the post class are set: tag_set, tag_set_predicted
    """
    assert isinstance(posts, list)

    recalls = []
    for post in posts:
        true_positive = 0
        false_negative = 0

        for tag in post.tag_set_prediction:
            if post.contains_tag_with_name(tag.name):
                true_positive += 1
        
        for tag1 in post.tag_set:
            exists = False
            for tag2 in post.tag_set_prediction:
                if tag1.name == tag2.name:
                    exists = True
                    break
            false_negative += 1 if not exists else 0

        r = float(true_positive) / float(true_positive + false_negative) if len(post.tag_set_prediction) > 0 else 0
        recalls.append(r)

    overall_precision = numpy.mean(recalls)
    return overall_precision
