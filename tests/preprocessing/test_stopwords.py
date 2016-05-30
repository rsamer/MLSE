import unittest
from entities.post import Post
from preprocessing import stopwords

class TestStopwords(unittest.TestCase):
    def test_stopwords(self):
        self.assert_stopword_removal("my house is small so I have to move out", "house I move")
        self.assert_stopword_removal("my name is To", "name To")
        self.assert_stopword_removal("my house was small", "house")
        self.assert_stopword_removal("my house tend to be small", "house tend")
        self.assert_stopword_removal("he she it was is to be from have had can can not should", "")
        self.assert_stopword_removal("actual this thing is also around anything", "thing")
        self.assert_stopword_removal("completely without yet", "")

    def assert_stopword_removal(self, body, expected_body):
        post = Post(1, "title", body, set([]), 1)
        post.tokens = body.split()

        stopwords.remove_stopwords([post])
        self.assertEqual(expected_body, " ".join(post.tokens))


if __name__ == '__main__':
    unittest.main()
