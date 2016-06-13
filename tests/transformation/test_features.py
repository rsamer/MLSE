import unittest
from entities.post import Post
from entities.tag import Tag
from transformation import features
from math import log, e


class TestFeatures(unittest.TestCase):
    def test_numeric_features(self):
        tag1 = Tag("tag1", 1)
        tag1.preprocessed_tag_name = "tag1"

        post1 = Post(1, "", "", {tag1}, 1)
        post1.title_tokens = ["title1", "tag1"]
        post1.body_tokens = ["body1", "ipsum", "lorem", "lorem"]

        post2 = Post(2, "", "", set(), 1)
        post2.title_tokens = ["title2", "tag2"]
        post2.body_tokens = ["body2", "ipsum", "lorem", "tag1"]

        train_posts = [post1]
        test_posts = [post2]
        tag_list = [tag1]
        X_train, X_test = features.numeric_features(train_posts, test_posts, tag_list)

        # probabilities
        p_tag = {"tag1": 1.0}
        p_title = {"title1": 0.5, "tag1": 0.5}
        p_tag_title = {"tag1_title1": 0.5, "tag1_tag1": 0.5}
        p_body = {"body1": 0.25, "ipsum": 0.25, "lorem": 0.5}
        p_tag_body = {"tag1_body1": 0.25, "tag1_ipsum": 0.25, "tag1_lorem": 0.5}

        # train data
        self.assertEqual(1, X_train.shape[0])
        self.assertEqual(4, X_train.shape[1])
        self.assertEqual(1, X_train[0, 0])  # title contains tag1
        self.assertEqual(0, X_train[0, 1])  # body contains tag1
        self._assertPmi(p_tag, p_title, p_tag_title, p_body, p_tag_body, post1, tag1, X_train[0, 2], title=True)  # title PMI tag1
        self._assertPmi(p_tag, p_title, p_tag_title, p_body, p_tag_body, post1, tag1, X_train[0, 3])  # body PMI tag1

        # test data
        self.assertEqual(1, X_test.shape[0])
        self.assertEqual(4, X_test.shape[1])
        self.assertEqual(0, X_test[0, 0])  # title contains tag1
        self.assertEqual(1, X_test[0, 1])  # body contains tag1
        self._assertPmi(p_tag, p_title, p_tag_title, p_body, p_tag_body, post2, tag1, X_test[0, 2], title=True)  # title PMI tag1
        self._assertPmi(p_tag, p_title, p_tag_title, p_body, p_tag_body, post2, tag1, X_test[0, 3])  # body PMI tag1

    def test_numeric_features_relaxed(self):
        tag1 = Tag("tag1", 1)
        tag1.preprocessed_tag_name = "tag1"

        tag2 = Tag("tag-two", 1)
        tag2.preprocessed_tag_name = "tag-two"

        post1 = Post(1, "", "", {tag1, tag2}, 1)
        post1.title_tokens = ["title1", "tag1", "tag", "tag", "two"]
        post1.body_tokens = ["body1", "ipsum", "lorem"]

        post2 = Post(2, "", "", {tag2}, 1)
        post2.title_tokens = ["title2", "tag2"]
        post2.body_tokens = ["body2", "ipsum", "lorem", "tag-two"]

        tag_list = [tag1, tag2]
        train_posts = [post1, post2]
        X_train, X_test = features.numeric_features(train_posts, [], tag_list)

        # probabilities
        p_tag = {"tag1": 1.0/3.0, "tag-two": 2.0/3.0}
        p_title = {"title1": 1.0/7.0, "tag1": 1.0/7.0, "tag": 2.0/7.0,
                   "two": 1.0/7.0, "title2": 1.0/7.0, "tag2": 1.0/7.0}
        p_tag_title = {"tag1_title1": 1.0/12.0, "tag1_tag1": 1.0/12.0, "tag1_tag": 2.0/12.0, "tag1_two": 1.0/12.0,
                       "tag-two_title1": 1.0/12.0, "tag-two_tag1": 1.0/12.0, "tag-two_tag": 2.0/12.0,
                       "tag-two_two": 1.0/12.0, "tag-two_title2": 1.0/12.0, "tag-two_tag2": 1.0/12.0}
        p_body = {"body1": 1.0/7.0, "ipsum": 2.0/7.0, "lorem": 2.0/7.0, "body2": 1.0/7.0, "tag-two": 1.0/7.0}
        p_tag_body = {"tag1_body1": 1.0/10.0, "tag1_ipsum": 1.0/10.0, "tag1_lorem": 1.0/10.0, "tag-two_body1": 1.0/10.0,
                      "tag-two_ipsum": 2.0/10.0, "tag-two_lorem": 2.0/10.0, "tag-two_body2": 1.0/10.0,
                      "tag-two_tag-two": 1.0/10.0}

        # train data
        self.assertEqual(2, X_train.shape[0])
        self.assertEqual(10, X_train.shape[1])

        # post1
        self.assertEqual(1, X_train[0, 0])  # title contains tag1
        self.assertEqual(0, X_train[0, 1])  # body contains tag1
        self._assertPmi(p_tag, p_title, p_tag_title, p_body, p_tag_body, post1, tag1, X_train[0, 2], title=True)  # title PMI tag1
        self._assertPmi(p_tag, p_title, p_tag_title, p_body, p_tag_body, post1, tag1, X_train[0, 3])  # body PMI tag1
        self.assertEqual(0, X_train[0, 4])  # title contains tag2
        self.assertEqual(0, X_train[0, 5])  # body contains tag2
        self.assertEqual(1, X_train[0, 6])  # title contains relaxed tag2
        self.assertEqual(0, X_train[0, 7])  # body contains relaxed tag2
        self._assertPmi(p_tag, p_title, p_tag_title, p_body, p_tag_body, post1, tag2, X_train[0, 8], title=True)  # title PMI tag2
        self._assertPmi(p_tag, p_title, p_tag_title, p_body, p_tag_body, post1, tag2, X_train[0, 9])  # body PMI tag2

        # post2
        self.assertEqual(0, X_train[1, 0])  # title contains tag1
        self.assertEqual(0, X_train[1, 1])  # body contains tag1
        self._assertPmi(p_tag, p_title, p_tag_title, p_body, p_tag_body, post2, tag1, X_train[1, 2], title=True)  # title PMI tag1
        self._assertPmi(p_tag, p_title, p_tag_title, p_body, p_tag_body, post2, tag1, X_train[1, 3])  # body PMI tag1
        self.assertEqual(0, X_train[1, 4])  # title contains tag2
        self.assertEqual(1, X_train[1, 5])  # body contains tag2
        self.assertEqual(0, X_train[1, 6])  # title contains relaxed tag2
        self.assertEqual(0, X_train[1, 7])  # body contains relaxed tag2
        self._assertPmi(p_tag, p_title, p_tag_title, p_body, p_tag_body, post2, tag2, X_train[1, 8], title=True)  # title PMI tag2
        self._assertPmi(p_tag, p_title, p_tag_title, p_body, p_tag_body, post2, tag2, X_train[1, 9])  # body PMI tag2

    def test_normalize_features(self):
        X = [[1.0, 0.0, 21.238], [2.0, -5.0, 0.1], [2.1, -10.4, 0.2]]
        min, max = self._get_min_max(X)
        self.assertEqual(-10.4, min)
        self.assertEqual(21.238, max)

        features._normalize_features(X)

        min, max = self._get_min_max(X)
        self.assertEqual(1.0, max)
        self.assertEqual(0.0, min)
        self.assertEqual((1.0 - -10.4) / (21.238 - -10.4), X[0][0])
        self.assertEqual((0.0 - -10.4) / (21.238 - -10.4), X[0][1])
        self.assertEqual((21.238 - -10.4) / (21.238 - -10.4), X[0][2])
        self.assertEqual((2.0 - -10.4) / (21.238 - -10.4), X[1][0])
        self.assertEqual((-5.0 - -10.4) / (21.238 - -10.4), X[1][1])
        self.assertEqual((0.1 - -10.4) / (21.238 - -10.4), X[1][2])
        self.assertEqual((2.1 - -10.4) / (21.238 - -10.4), X[2][0])
        self.assertEqual((-10.4 - -10.4) / (21.238 - -10.4), X[2][1])
        self.assertEqual((0.2 - -10.4) / (21.238 - -10.4), X[2][2])

    def _assertPmi(self, p_tag, p_title, p_tag_title, p_body, p_tag_body, post, tag, pmi, title=False):
        expected_pmi = 0.0
        p_tag_token = p_tag_title if title else p_tag_body
        p_token = p_title if title else p_body

        for token in set((post.title_tokens if title else post.body_tokens)):
            tag_token = "_".join([tag.name, token])
            if tag_token in p_tag_token:
                expected_pmi += log(p_tag_token[tag_token] / (p_tag[tag.name] * p_token[token]), e)

        self.assertEqual(expected_pmi, pmi)

    def _get_min_max(self, X):
        minimum = None
        maximum = None
        for row in X:
            for value in row:
                if minimum is None or value < minimum:
                    minimum = value
                if maximum is None or value > maximum:
                    maximum = value
        return minimum, maximum


if __name__ == '__main__':
    unittest.main()
