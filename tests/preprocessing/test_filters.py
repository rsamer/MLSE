import unittest
from entities.post import Post
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

        post = Post(1, "title", "this is &nbsp;", set([]), 1)
        filters.strip_html_tags([post])
        self.assertEqual("this is", post.body)


if __name__ == '__main__':
    unittest.main()
