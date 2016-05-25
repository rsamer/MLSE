import unittest
from entities.post import Post
from preprocessing import tokenizer

class TestTokenizer(unittest.TestCase):
    def test_tokenizer(self):
        self.assert_tokens("this is a test", [], ["this", "is", "a", "test"])

        # TODO do not concatenate title to body in tokenize_posts and continue here...

    def assert_tokens(self, body, tags, expected_tokens):
        post = Post(1, "title", body, set([]), 1)
        post.tokens = body.split()

        tokenizer.tokenize_posts([post], tag_names=tags)
        self.assertEqual(expected_tokens, post.tokens)


if __name__ == '__main__':
    unittest.main()
