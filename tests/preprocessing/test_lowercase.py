import unittest
from entities.post import Post
from preprocessing import filters

class TestToLowercase(unittest.TestCase):
    def test_tokenizer(self):
        self.assert_tokens("this is a test", "this is a test")
        self.assert_tokens("this is A Test", "this is a test")
        self.assert_tokens("C++", "c++")
        self.assert_tokens("test!", "test!")

    def assert_tokens(self, body, expected_body):
        title = "TITLE"
        post = Post(1, title, body, set([]), 1)
        post.tokens = body.split()

        filters.to_lower_case([post])
        self.assertEqual(expected_body, post.body)
        self.assertEqual(title.lower(), post.title)


if __name__ == '__main__':
    unittest.main()
