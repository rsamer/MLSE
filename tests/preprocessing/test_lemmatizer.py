import unittest
from entities.post import Post
from preprocessing import pos
from preprocessing import lemmatizer


class TestLemmatizer(unittest.TestCase):
    def test_lemmatizer(self):
        post = Post(1, "", "", set([]), 1)

        post.title_tokens = ["writing", "tested"]
        post.body_tokens = ["object", "oriented", "object-oriented"]

        pos.pos_tagging([post])
        lemmatizer.word_net_lemmatizer([post])

        self.assertEqual(["write", "test"], post.title_tokens)
        self.assertEqual(["object", "orient", "object-oriented"], post.body_tokens)

if __name__ == '__main__':
    unittest.main()
