import unittest
from entities.post import Post
from preprocessing import stopwords


class TestStopwords(unittest.TestCase):
    def test_stopwords(self):
        self.assert_stopword_removal("my house is small so I have to move out", "house I move", [])
        self.assert_stopword_removal("my name is To", "name To", [])
        self.assert_stopword_removal("my house was small", "house", [])
        self.assert_stopword_removal("my house tend to be small", "house tend", [])
        self.assert_stopword_removal("he she it was is to be from have had can can not should", "", [])
        self.assert_stopword_removal("actual this thing is also around anything", "thing", [])
        self.assert_stopword_removal("completely without yet", "", [])

    def test_donot_remove_stopwords_that_are_tag_names(self):
        self.assert_stopword_removal("my house is small so I have to move out", "house so I move", ["so"])
        self.assert_stopword_removal("actual this thing is also around anything", "actual thing anything", ["actual", "anything"])
        self.assert_stopword_removal("completely without yet", "completely", ["completely"])

    def assert_stopword_removal(self, body, expected_body, tag_names):
        post = Post(1, "title my is to", body, set([]), 1)
        post.title_tokens = post.title.split()
        post.body_tokens = post.body.split()

        stopwords.remove_stopwords([post], tag_names)
        self.assertEqual("title", " ".join(post.title_tokens))
        self.assertEqual(expected_body, " ".join(post.body_tokens))


if __name__ == '__main__':
    unittest.main()
