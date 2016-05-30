import unittest
from entities.post import Post
from entities.post import Answer
from preprocessing import filters


class TestFilters(unittest.TestCase):
    def test_to_lower_case(self):
        post = Post(1, "TITLE", "this is a test", set([]), 1)
        filters.to_lower_case([post])
        self.assertEqual("this is a test", post.body)
        self.assertEqual("title", post.title)

        post = Post(1, "title", "THIS is a Test", set([]), 1)
        filters.to_lower_case([post])
        self.assertEqual("this is a test", post.body)
        self.assertEqual("title", post.title)

    def test_strip_code_segments(self):
        post = Post(1, "title", "this is a <code>if else exit</code> test", set([]), 1)
        filters.strip_code_segments([post])
        self.assertEqual("this is a test", post.body)

        post = Post(1, "title", "this is a<code>if else exit</code>test", set([]), 1)
        filters.strip_code_segments([post])
        self.assertEqual("this is a test", post.body)

    def test_strip_html_tags(self):
        post = Post(1, "title", "this is a <strong>test</strong>", set([]), 1)
        filters.strip_html_tags([post])
        self.assertEqual("this is a test", post.body)

        post = Post(1, "title", "<html>this is a <strong>test</strong></html>", set([]), 1)
        filters.strip_html_tags([post])
        self.assertEqual("this is a test", post.body)

        post = Post(1, "title", "<html>this is a <strong>test</html>", set([]), 1)
        filters.strip_html_tags([post])
        self.assertEqual("this is a test", post.body)

        post = Post(1, "title", "this is a <img /> test", set([]), 1)
        filters.strip_html_tags([post])
        self.assertEqual("this is a  test", post.body)

        post = Post(1, "title", "this is &nbsp; a test", set([]), 1)
        filters.strip_html_tags([post])
        self.assertEqual("this is   a test", post.body)

    def test_tokens(self):
        post = Post(1, "", "", set([]), 1)

        post.tokens = ["www.tugraz.at", "http://www.tugraz.at", "test"]
        filters.filter_tokens([post], [])
        self.assertEqual(["test"], post.tokens)

        post.tokens = [":-)", ":D", ";)", "xD", ":o)", "test"]
        filters.filter_tokens([post], [])
        self.assertEqual(["test"], post.tokens)

        post.tokens = ["_test_", "a", "bc"]
        filters.filter_tokens([post], [])
        self.assertEqual([], post.tokens)

        post.tokens = ["a", "bc"]
        filters.filter_tokens([post], ["a"])
        self.assertEqual(["a"], post.tokens)

        post.tokens = ["he's", "planets'", "you're", "we've", "isn't", "haven't"]
        filters.filter_tokens([post], [])
        self.assertEqual(["he", "planet", "you", "we"], post.tokens)

        post.tokens = ["#123", "#124.2", "123,45", "123.45", "#FF0000", "#ff0000", "test"]
        filters.filter_tokens([post], [])
        self.assertEqual(["test"], post.tokens)

    def test_filter_less_relevant_posts(self):
        post1 = Post(1, "", "", set([]), 10)
        post2 = Post(2, "", "", set([]), 11)

        posts = [post1, post2]
        posts = filters.filter_less_relevant_posts(posts, 11)
        self.assertEqual([post2], posts)

        post2.score = 0
        posts = [post1, post2]
        posts = filters.filter_less_relevant_posts(posts, -1)
        self.assertEqual([post1], posts)

        post2.score = 10
        posts = [post1, post2]
        posts = filters.filter_less_relevant_posts(posts, 2)
        self.assertEqual([post1, post2], posts)

        post1.score = 10
        post2.score = 0
        post2.answers = [Answer(3, "body", 1)]
        posts = [post1, post2]
        posts = filters.filter_less_relevant_posts(posts, 0)
        self.assertEqual([post1, post2], posts)

if __name__ == '__main__':
    unittest.main()
