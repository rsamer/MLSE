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


if __name__ == '__main__':
    unittest.main()
