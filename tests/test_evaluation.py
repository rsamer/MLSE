import unittest
from entities.post import Post
from entities.tag import Tag
from unsupervised import evaluation


class TestEvaluation(unittest.TestCase):
    def test_precision_no_false_positives(self):
        tag1 = Tag("tag1", 1)
        tag2 = Tag("tag2", 1)

        post = Post(1, "title", "body", set([tag1, tag2]), 1)
        post.tag_set_prediction = set([tag1])

        precision = evaluation.precision([post])
        self.assertEqual(1.0 / (1.0 + 0.0), precision)

    def test_precision_no_positives(self):
        tag1 = Tag("tag1", 1)
        tag2 = Tag("tag2", 1)

        post = Post(1, "title", "body", set([tag1, tag2]), 1)
        post.tag_set_prediction = set()

        precision = evaluation.precision([post])
        self.assertEqual(0.0, precision)

    def test_precision_no_true_positives(self):
        tag1 = Tag("tag1", 1)
        tag2 = Tag("tag2", 1)

        post = Post(1, "title", "body", set([tag1]), 1)
        post.tag_set_prediction = set([tag2])

        precision = evaluation.precision([post])
        self.assertEqual(0.0, precision)

    def test_precision(self):
        tag1 = Tag("tag1", 1)
        tag2 = Tag("tag2", 1)
        tag3 = Tag("tag3", 1)
        tag4 = Tag("tag4", 1)

        post = Post(1, "title", "body", set([tag1, tag2]), 1)
        post.tag_set_prediction = set([tag2, tag3, tag4])

        precision = evaluation.precision([post])
        self.assertEqual(1.0 / (1.0 + 2.0), precision)


if __name__ == '__main__':
    unittest.main()
