import unittest

from entities.post import Post
from entities.tag import Tag
from evaluation import evaluation


class TestRecall(unittest.TestCase):
    def test_recall_no_false_negatives(self):
        tag1 = Tag("tag1", 1)
        tag2 = Tag("tag2", 1)

        post = Post(1, "title", "body", set([tag1, tag2]), 1)
        post.tag_set_prediction = set([tag1, tag2])

        recall = evaluation.recall([post])
        self.assertEqual(1.0 / (1.0 + 0.0), recall)

    def test_recall_no_positives(self):
        tag1 = Tag("tag1", 1)
        tag2 = Tag("tag2", 1)
        tag3 = Tag("tag3", 1)

        post = Post(1, "title", "body", set([tag1, tag2]), 1)
        post.tag_set_prediction = set([tag3])

        recall = evaluation.recall([post])
        self.assertEqual(0.0 / (0.0 + 2.0), recall)

    def test_recall(self):
        tag1 = Tag("tag1", 1)
        tag2 = Tag("tag2", 1)
        tag3 = Tag("tag3", 1)
        tag4 = Tag("tag4", 1)

        post = Post(1, "title", "body", set([tag1, tag2]), 1)
        post.tag_set_prediction = set([tag2, tag3, tag4])

        recall = evaluation.recall([post])
        self.assertEqual(1.0 / (1.0 + 1.0), recall)


if __name__ == '__main__':
    unittest.main()
