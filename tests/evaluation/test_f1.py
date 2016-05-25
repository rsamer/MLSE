import unittest

from entities.post import Post
from entities.tag import Tag
from evaluation import evaluation


class TestF1(unittest.TestCase):
    def test_f1_no_false_positives(self):
        tag1 = Tag("tag1", 1)
        tag2 = Tag("tag2", 1)

        post = Post(1, "title", "body", set([tag1, tag2]), 1)
        post.tag_set_prediction = set([tag1])

        precision = evaluation.precision([post])
        recall = evaluation.recall([post])
        f1 = evaluation.f1([post])
        self.assertEqual(2.0 * precision * recall / (precision + recall), f1)

        expected_precision = 1.0
        expected_recall = 0.5
        self.assertEqual(2.0 * expected_precision * expected_recall / (expected_precision + expected_recall), f1)

    def test_f1_no_positives(self):
        tag1 = Tag("tag1", 1)
        tag2 = Tag("tag2", 1)

        post = Post(1, "title", "body", set([tag1, tag2]), 1)
        post.tag_set_prediction = set()

        f1 = evaluation.f1([post])
        self.assertEqual(0.0, f1)

    def test_f1_no_true_positives(self):
        tag1 = Tag("tag1", 1)
        tag2 = Tag("tag2", 1)

        post = Post(1, "title", "body", set([tag1]), 1)
        post.tag_set_prediction = set([tag2])

        f1 = evaluation.f1([post])
        self.assertEqual(0.0, f1)

    def test_f1(self):
        tag1 = Tag("tag1", 1)
        tag2 = Tag("tag2", 1)
        tag3 = Tag("tag3", 1)
        tag4 = Tag("tag4", 1)

        post = Post(1, "title", "body", set([tag1, tag2]), 1)
        post.tag_set_prediction = set([tag2, tag3, tag4])

        f1 = evaluation.f1([post])

        expected_precision = 0.33
        expected_recall = 0.5
        self.assertAlmostEqual(2.0 * expected_precision * expected_recall / (expected_precision + expected_recall), f1, delta=0.01)


if __name__ == '__main__':
    unittest.main()
