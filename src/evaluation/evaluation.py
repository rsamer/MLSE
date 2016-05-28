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


def f1(posts):
    """
    Computes the precision for a tag prediction.
    Make sure these attributes of the post class are set: tag_set, tag_set_predicted
    """
    assert isinstance(posts, list)
    p = precision(posts)
    r = recall(posts)
    return (2.0 * p * r) / (p + r) if (p + r) > 0 else 0


def print_evaluation_results(test_posts):
    print "Overall precision = %0.3f" % precision(test_posts)
    print "Overall recall = %0.3f" % recall(test_posts)
    print "Overall f1 = %0.3f" % f1(test_posts)


def print_tag_evaluation(posts, tags):
    """
    NOTE: used for testing!
    Computes the precision and accuracy for a specific tag overall posts.
    Make sure these attributes of the post class are set: tag_set, tag_set_predicted
    """
    assert isinstance(posts, list)

    for tag in tags[0:5]:
        true_positive = 0
        false_positive = 0
        false_negative = 0

        for post in posts:
            exists_in_tag_set = post.contains_tag_with_name(tag.name)
            exists_in_tag_prediction_set = False

            for tag_predicted in post.tag_set_prediction:
                if tag_predicted.name == tag.name:
                    exists_in_tag_prediction_set = True
                    break

            if not exists_in_tag_set and not exists_in_tag_prediction_set:
                continue

            true_positive += 1 if exists_in_tag_set and exists_in_tag_prediction_set else 0
            false_positive += 1 if exists_in_tag_prediction_set and not exists_in_tag_set else 0
            false_negative += 1 if exists_in_tag_set and not exists_in_tag_prediction_set else 0

        p = (float(true_positive) / float(true_positive + false_positive))
        r = (float(true_positive) / float(true_positive + false_negative))
        print "Evaluation for tag '" + tag.name + "': Precision=" + str(p) + ", Recall=" + str(r)
