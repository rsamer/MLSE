import unittest
from entities.post import Post
from transformation import tfidf
from math import log, e


class TestTransformation(unittest.TestCase):
    def test_tf_idf_single_occurrence(self):
        post1 = Post(1, "", "", set([]), 1)
        post1.title_tokens = ["title1"]
        post1.body_tokens = ["body1", "ipsum", "lorem"]

        post2 = Post(1, "", "", set([]), 1)
        post2.title_tokens = ["title2"]
        post2.body_tokens = ["body2", "ipsum", "lorem"]

        post3 = Post(1, "", "", set([]), 1)
        post3.title_tokens = ["title3"]
        post3.body_tokens = ["body3", "ipsum", "lorem"]

        train_posts = [post1, post2]
        test_posts = [post3]

        X_train, X_test = tfidf.tfidf(train_posts, test_posts, max_features=None, min_df=1, norm=None)
        n_docs = float(len(train_posts))

        # train data
        X_train_dense = X_train.todense()
        self.assertEqual(2, X_train_dense.shape[0])
        self.assertEqual(6, X_train_dense.shape[1])

        # post1
        self.assert_tf_idf(X_train_dense[0, 0], 1.0, 1.0, n_docs) # body1
        self.assert_tf_idf(X_train_dense[0, 1], 0.0, 1.0, n_docs) # body2
        self.assert_tf_idf(X_train_dense[0, 2], 1.0, 2.0, n_docs) # ipsum
        self.assert_tf_idf(X_train_dense[0, 3], 1.0, 2.0, n_docs) # lorem
        self.assert_tf_idf(X_train_dense[0, 4], 3.0, 1.0, n_docs)  # title1
        self.assert_tf_idf(X_train_dense[0, 5], 0.0, 1.0, n_docs)  # title2

        # post2
        self.assert_tf_idf(X_train_dense[1, 0], 0.0, 1.0, n_docs)  # body1
        self.assert_tf_idf(X_train_dense[1, 1], 1.0, 1.0, n_docs)  # body2
        self.assert_tf_idf(X_train_dense[1, 2], 1.0, 2.0, n_docs)  # ipsum
        self.assert_tf_idf(X_train_dense[1, 3], 1.0, 2.0, n_docs)  # lorem
        self.assert_tf_idf(X_train_dense[1, 4], 0.0, 1.0, n_docs)  # title1
        self.assert_tf_idf(X_train_dense[1, 5], 3.0, 1.0, n_docs)  # title2

        # test data
        X_test_dense = X_test.todense()
        self.assertEqual(1, X_test_dense.shape[0])
        self.assertEqual(6, X_test_dense.shape[1])

        # post3
        self.assert_tf_idf(X_test_dense[0, 0], 0.0, 1.0, n_docs)  # body1
        self.assert_tf_idf(X_test_dense[0, 1], 0.0, 1.0, n_docs)  # body2
        self.assert_tf_idf(X_test_dense[0, 2], 1.0, 2.0, n_docs)  # ipsum
        self.assert_tf_idf(X_test_dense[0, 3], 1.0, 2.0, n_docs)  # lorem
        self.assert_tf_idf(X_test_dense[0, 4], 0.0, 1.0, n_docs)  # title1
        self.assert_tf_idf(X_test_dense[0, 5], 0.0, 1.0, n_docs)  # title2

    def test_tf_idf_multiple_occurrences(self):
        post1 = Post(1, "", "", set([]), 1)
        post1.title_tokens = []
        post1.body_tokens = ["body1", "ipsum", "lorem", "lorem", "lorem"]

        post2 = Post(1, "", "", set([]), 1)
        post2.title_tokens = []
        post2.body_tokens = ["body2", "ipsum", "lorem"]

        post3 = Post(1, "", "", set([]), 1)
        post3.title_tokens = []
        post3.body_tokens = ["body3", "ipsum", "lorem", "ipsum"]

        train_posts = [post1, post2]
        test_posts = [post3]

        X_train, X_test = tfidf.tfidf(train_posts, test_posts, max_features=None, min_df=1, norm=None)
        n_docs = float(len(train_posts))

        # train data
        X_train_dense = X_train.todense()
        self.assertEqual(2, X_train_dense.shape[0])
        self.assertEqual(4, X_train_dense.shape[1])

        # post1
        self.assert_tf_idf(X_train_dense[0, 0], 1.0, 1.0, n_docs)  # body1
        self.assert_tf_idf(X_train_dense[0, 1], 0.0, 1.0, n_docs)  # body2
        self.assert_tf_idf(X_train_dense[0, 2], 1.0, 2.0, n_docs)  # ipsum
        self.assert_tf_idf(X_train_dense[0, 3], 3.0, 2.0, n_docs)  # lorem

        # post2
        self.assert_tf_idf(X_train_dense[1, 0], 0.0, 1.0, n_docs)  # body1
        self.assert_tf_idf(X_train_dense[1, 1], 1.0, 1.0, n_docs)  # body2
        self.assert_tf_idf(X_train_dense[1, 2], 1.0, 2.0, n_docs)  # ipsum
        self.assert_tf_idf(X_train_dense[1, 3], 1.0, 2.0, n_docs)  # lorem

        # test data
        X_test_dense = X_test.todense()
        self.assertEqual(1, X_test_dense.shape[0])
        self.assertEqual(4, X_test_dense.shape[1])

        # post3
        self.assert_tf_idf(X_test_dense[0, 0], 0.0, 1.0, n_docs)  # body1
        self.assert_tf_idf(X_test_dense[0, 1], 0.0, 1.0, n_docs)  # body2
        self.assert_tf_idf(X_test_dense[0, 2], 2.0, 2.0, n_docs)  # ipsum
        self.assert_tf_idf(X_test_dense[0, 3], 1.0, 2.0, n_docs)  # lorem

    def test_tf_idf_min_df(self):
        post1 = Post(1, "", "", set([]), 1)
        post1.title_tokens = []
        post1.body_tokens = ["body1", "ipsum", "lorem", "lorem", "lorem"]

        post2 = Post(1, "", "", set([]), 1)
        post2.title_tokens = []
        post2.body_tokens = ["body2", "ipsum", "lorem"]

        post3 = Post(1, "", "", set([]), 1)
        post3.title_tokens = []
        post3.body_tokens = ["body3", "ipsum", "lorem", "ipsum"]

        train_posts = [post1, post2]
        test_posts = [post3]

        X_train, X_test = tfidf.tfidf(train_posts, test_posts, max_features=None, min_df=2, norm=None)
        n_docs = float(len(train_posts))

        # train data
        X_train_dense = X_train.todense()
        self.assertEqual(2, X_train_dense.shape[0])
        self.assertEqual(2, X_train_dense.shape[1])

        # post1
        self.assert_tf_idf(X_train_dense[0, 0], 1.0, 2.0, n_docs)  # ipsum
        self.assert_tf_idf(X_train_dense[0, 1], 3.0, 2.0, n_docs)  # lorem

        # post2
        self.assert_tf_idf(X_train_dense[1, 0], 1.0, 2.0, n_docs)  # ipsum
        self.assert_tf_idf(X_train_dense[1, 1], 1.0, 2.0, n_docs)  # lorem

        # test data
        X_test_dense = X_test.todense()
        self.assertEqual(1, X_test_dense.shape[0])
        self.assertEqual(2, X_test_dense.shape[1])

        # post3
        self.assert_tf_idf(X_test_dense[0, 0], 2.0, 2.0, n_docs)  # ipsum
        self.assert_tf_idf(X_test_dense[0, 1], 1.0, 2.0, n_docs)  # lorem

    def test_tf_idf_max_features(self):
        post1 = Post(1, "", "", set([]), 1)
        post1.title_tokens = []
        post1.body_tokens = ["body1", "ipsum", "lorem", "lorem", "lorem"]

        post2 = Post(1, "", "", set([]), 1)
        post2.title_tokens = []
        post2.body_tokens = ["body2", "ipsum", "lorem"]

        post3 = Post(1, "", "", set([]), 1)
        post3.title_tokens = []
        post3.body_tokens = ["body3", "ipsum", "lorem", "ipsum"]

        train_posts = [post1, post2]
        test_posts = [post3]

        X_train, X_test = tfidf.tfidf(train_posts, test_posts, max_features=1, min_df=1, norm=None)
        n_docs = float(len(train_posts))

        # train data
        X_train_dense = X_train.todense()
        self.assertEqual(2, X_train_dense.shape[0])
        self.assertEqual(1, X_train_dense.shape[1])

        # post1
        self.assert_tf_idf(X_train_dense[0, 0], 3.0, 2.0, n_docs)  # lorem

        # post2
        self.assert_tf_idf(X_train_dense[1, 0], 1.0, 2.0, n_docs)  # lorem

        # test data
        X_test_dense = X_test.todense()
        self.assertEqual(1, X_test_dense.shape[0])
        self.assertEqual(1, X_test_dense.shape[1])

        # post3
        self.assert_tf_idf(X_test_dense[0, 0], 1.0, 2.0, n_docs)  # lorem

    def assert_tf_idf(self, tf_idf, term_frequency_in_doc, n_docs_with_term, n_docs):
        # The actual formula used for tf-idf is tf * (idf + 1) = tf + tf * idf,
        # instead of tf * idf. The effect of this is that terms with zero idf, i.e.
        # that occur in all documents of a training set, will not be entirely ignored.
        # see text.py of sklearn
        self.assertEqual(tf_idf, term_frequency_in_doc * (log((n_docs + 1) / (n_docs_with_term + 1), e) + 1.0))


if __name__ == '__main__':
    unittest.main()
