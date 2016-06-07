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
        post = Post(1, "title &nbsp;", "this is a <strong>test</strong>", set([]), 1)
        filters.strip_html_tags([post])
        self.assertEqual("title", post.title)
        self.assertEqual("this is a test", post.body)

        post = Post(1, "<strong>title</strong>", "<html>this is a <strong>test</strong></html>", set([]), 1)
        filters.strip_html_tags([post])
        self.assertEqual("title", post.title)
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

    def test_filter_tokens(self):
        post = Post(1, "", "", set([]), 1)

        post.title_tokens = ["www.tugraz.at", "title"]
        post.body_tokens = ["www.tugraz.at", "http://www.tugraz.at", "test"]
        filters.filter_tokens([post], [])
        self.assertEqual(["title"], post.title_tokens)
        self.assertEqual(["test"], post.body_tokens)

        post.body_tokens = [":-)", ":D", ";)", "xD", ":o)", "test"]
        filters.filter_tokens([post], [])
        self.assertEqual(["test"], post.body_tokens)

        post.body_tokens = ["_test_", "a", "bc"]
        filters.filter_tokens([post], [])
        self.assertEqual([], post.body_tokens)

        post.title_tokens = ["a", "test"]
        post.body_tokens = ["a", "bc"]
        filters.filter_tokens([post], ["a"])
        self.assertEqual(["a", "test"], post.title_tokens)
        self.assertEqual(["a"], post.body_tokens)
        self.assertEqual(["a", "test", "a"], post.tokens(1))

        post.body_tokens = ["he's", "planets'", "you're", "we've", "isn't", "haven't"]
        filters.filter_tokens([post], [])
        self.assertEqual(["he", "planet", "you", "we"], post.body_tokens)

        post.body_tokens = ["#123", "#124.2", "123,45", "123.45", "1,234.56", "#FF0000", "#ff0000", "test"]
        filters.filter_tokens([post], [])
        self.assertEqual(["123,45", "123.45", "1,234.56", "test"], post.body_tokens)

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
