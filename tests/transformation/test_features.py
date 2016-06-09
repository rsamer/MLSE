import unittest
from entities.post import Post
from entities.tag import Tag
from transformation import features
from math import log


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

        # train data
        self.assertEqual(1, X_train.shape[0])
        self.assertEqual(4, X_train.shape[1])
        self.assertEqual(1, X_train[0, 0])  # title contains tag1
        self.assertEqual(0, X_train[0, 1])  # body contains tag1
        self.assertPmi(train_posts, tag_list, post1, tag1, X_train[0, 2], title=True)  # title PMI tag1
        self.assertPmi(train_posts, tag_list, post1, tag1, X_train[0, 3])  # body PMI tag1

        # test data
        self.assertEqual(1, X_test.shape[0])
        self.assertEqual(4, X_test.shape[1])
        self.assertEqual(0, X_test[0, 0])  # title contains tag1
        self.assertEqual(1, X_test[0, 1])  # body contains tag1
        self.assertPmi(train_posts, tag_list, post2, tag1, X_test[0, 2], title=True)  # title PMI tag1
        self.assertPmi(train_posts, tag_list, post2, tag1, X_test[0, 3])  # body PMI tag1

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

        # train data
        self.assertEqual(2, X_train.shape[0])
        self.assertEqual(10, X_train.shape[1])

        # post1
        self.assertEqual(1, X_train[0, 0])  # title contains tag1
        self.assertEqual(0, X_train[0, 1])  # body contains tag1
        self.assertPmi(train_posts, tag_list, post1, tag1, X_train[0, 2], title=True)  # title PMI tag1
        self.assertPmi(train_posts, tag_list, post1, tag1, X_train[0, 3])  # body PMI tag1
        self.assertEqual(0, X_train[0, 4])  # title contains tag2
        self.assertEqual(0, X_train[0, 5])  # body contains tag2
        self.assertEqual(1, X_train[0, 6])  # title contains relaxed tag2
        self.assertEqual(0, X_train[0, 7])  # body contains relaxed tag2
        self.assertPmi(train_posts, tag_list, post1, tag2, X_train[0, 8], title=True)  # title PMI tag2
        self.assertPmi(train_posts, tag_list, post1, tag2, X_train[0, 9])  # body PMI tag2

        # post2
        self.assertEqual(0, X_train[1, 0])  # title contains tag1
        self.assertEqual(0, X_train[1, 1])  # body contains tag1
        self.assertPmi(train_posts, tag_list, post2, tag1, X_train[1, 2], title=True)  # title PMI tag1
        self.assertPmi(train_posts, tag_list, post2, tag1, X_train[1, 3])  # body PMI tag1
        self.assertEqual(0, X_train[1, 4])  # title contains tag2
        self.assertEqual(1, X_train[1, 5])  # body contains tag2
        self.assertEqual(0, X_train[1, 6])  # title contains relaxed tag2
        self.assertEqual(0, X_train[1, 7])  # body contains relaxed tag2
        self.assertPmi(train_posts, tag_list, post2, tag2, X_train[1, 8], title=True)  # title PMI tag2
        self.assertPmi(train_posts, tag_list, post2, tag2, X_train[1, 9])  # body PMI tag2

    def assertPmi(self, train_posts, tag_list, post, tag, pmi, title=False):
        expected_pmi = 0.0
        p_tag = 0.0
        p_token_tag = {}
        p_token = {}
        tag_token_assignment = {}

        for p in train_posts:
            for t in p.tag_set:
                if t.name == tag.name:
                    p_tag += 1.0

            for t in (p.title_tokens if title else p.body_tokens):
                if t not in p_token:
                    p_token[t] = 0.0
                p_token[t] += 1.0

                for ta in p.tag_set:
                    tag_token_assignment["_".join([ta.name, t])] = True

                if tag in p.tag_set:
                    if t not in p_token_tag:
                        p_token_tag[t] = 0.0
                    p_token_tag[t] += 1.0

        p_tag /= len(tag_list)

        for p_tt in p_token_tag:
            p_token_tag[p_tt] /= len(tag_token_assignment)

        for p_t in p_token:
            p_token[p_t] /= len(p_token)

        for t in (post.title_tokens if title else post.body_tokens):
            if t in p_token_tag:
                expected_pmi += log(p_token_tag[t] / (p_tag * p_token[t]))

        self.assertEqual(expected_pmi, pmi)


if __name__ == '__main__':
    unittest.main()
