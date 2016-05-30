import unittest
from entities.post import Post
from preprocessing import tokenizer


class TestTokenizer(unittest.TestCase):
    def test_tokenizer(self):
        self.assert_tokens("this is a test", [], ["this", "is", "a", "test"])
        self.assert_tokens("this, is a test", [], ["this", "is", "a", "test"])
        self.assert_tokens("ASP.net is a programming-language", [], ["ASP.net", "is", "a", "programming-language"])
        self.assert_tokens("C++", [], ["C", "+", "+"])
        self.assert_tokens("C++", ["C++"], ["C++"])
        self.assert_tokens("C++, is a programming language", ["C++"], ["C++", "is", "a", "programming", "language"])
        self.assert_tokens("first.second", [], ["first.second"])
        self.assert_tokens("first. second", [], ["first", "second"])
        self.assert_tokens("tag-Name", [], ["tag-Name"])
        self.assert_tokens("tag-Name!", [], ["tag-Name"])
        self.assert_tokens("this ! is a test", [], ["this", "is", "a", "test"])
        self.assert_tokens("this . is a test", [], ["this", "is", "a", "test"])
        self.assert_tokens("this , is a test", [], ["this", "is", "a", "test"])
        self.assert_tokens("this ) is a test", [], ["this", "is", "a", "test"])
        self.assert_tokens("(this) is a test", [], ["this", "is", "a", "test"])
        self.assert_tokens("this: is a test", [], ["this", "is", "a", "test"])
        self.assert_tokens("this - is a test", [], ["this", "is", "a", "test"])
        self.assert_tokens("this? is a test", [], ["this", "is", "a", "test"])
        self.assert_tokens("don't", [], ["don't"])
        self.assert_tokens("do not", [], ["do", "not"])
        self.assert_tokens("hello 1234", [], ["hello", "1234"])

    def assert_tokens(self, body, tags, expected_tokens):
        post = Post(1, "title", body, set([]), 1)

        tokenizer.tokenize_posts([post], tag_names=tags)
        self.assertEqual(expected_tokens, post.tokens)


if __name__ == '__main__':
    unittest.main()
