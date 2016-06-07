import unittest

from entities.post import Post
from entities.tag import Tag
from evaluation import metrics


class TestF1(unittest.TestCase):

    def test_f1_no_false_positives(self):
        tag1 = Tag("tag1", 1)
        tag2 = Tag("tag2", 1)

        post = Post(1, "title", "body", set([tag1, tag2]), 1)
        post.tag_set_prediction = set([tag1])

        precision = metrics.precision([post])
        recall = metrics.recall([post])
        f1 = metrics.f1([post])
        self.assertEqual(2.0 * precision * recall / (precision + recall), f1)

        expected_precision = 1.0
        expected_recall = 0.5
        self.assertEqual(2.0 * expected_precision * expected_recall / (expected_precision + expected_recall), f1)

    def test_f1_no_positives(self):
        tag1 = Tag("tag1", 1)
        tag2 = Tag("tag2", 1)

        post = Post(1, "title", "body", set([tag1, tag2]), 1)
        post.tag_set_prediction = set()

        f1 = metrics.f1([post])
        self.assertEqual(0.0, f1)

    def test_f1_no_true_positives(self):
        tag1 = Tag("tag1", 1)
        tag2 = Tag("tag2", 1)

        post = Post(1, "title", "body", set([tag1]), 1)
        post.tag_set_prediction = set([tag2])

        f1 = metrics.f1([post])
        self.assertEqual(0.0, f1)

    def test_f1(self):
        tag1 = Tag("tag1", 1)
        tag2 = Tag("tag2", 1)
        tag3 = Tag("tag3", 1)
        tag4 = Tag("tag4", 1)

        post = Post(1, "title", "body", set([tag1, tag2]), 1)
        post.tag_set_prediction = set([tag2, tag3, tag4])

        f1 = metrics.f1([post])

        expected_precision = 0.33
        expected_recall = 0.5
        self.assertAlmostEqual(2.0 * expected_precision * expected_recall / (expected_precision + expected_recall), f1, delta=0.01)


    def test_f1_average_test(self):
        # based on: https://www.kaggle.com/wiki/MeanFScore
#         y_true = [[1, 2],
#                   [3, 4, 5],
#                   [6],
#                   [7]]
#         y_pred = [[1, 2, 3, 9],
#                   [3, 4],
#                   [6, 12],
#                   [1]]
#         
#         p1 = 2/(2+0) = 1
#         r1 = 2/(2+2) = 1/2
#         
#         p2 = 2/(2+1) = 2/3
#         r2 = 2/(2+0) = 1
#         
#         p3 = 1/(1+0) = 1
#         r3 = 1/(1+1) = 1/2
#         
#         p4 = 0
#         r4 = 0
#         
#         p_avg = (1.0+0.666666666+1.0+0.0)/4.0 = 0.6666666665
#         r_avg = (0.5+1.0+0.5+0.0)/4.0 = 0.5
#         
#         2.0*0.6666666665*0.5/(0.6666666665+0.5) = 0.571428571367

        tag1 = Tag("tag1", 1)
        tag2 = Tag("tag2", 1)
        tag3 = Tag("tag3", 1)
        tag4 = Tag("tag4", 1)
        tag5 = Tag("tag5", 1)
        tag6 = Tag("tag6", 1)
        tag7 = Tag("tag7", 1)
        tag9 = Tag("tag9", 1)
        tag12 = Tag("tag12", 1)

        y_true = [[tag1, tag2],
                  [tag3, tag4, tag5],
                  [tag6],
                  [tag7]]
        y_pred = [[tag1, tag2, tag3, tag9],
                  [tag3, tag4],
                  [tag6, tag12],
                  [tag1]]

        posts = []
        for idx, tags in enumerate(y_true):
            post = Post(idx, "title", "body", set(tags), 1)
            post.tag_set_prediction = set(y_pred[idx])
            posts.append(post)

        f1 = metrics.f1(posts)
        #self.assertAlmostEqual(0.53333333, f1, delta=0.01)
        self.assertAlmostEqual(0.571429, f1, delta=0.01)


if __name__ == '__main__':
    unittest.main()
