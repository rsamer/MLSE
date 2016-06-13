import unittest
import os
import main
from entities.post import Post
from entities.tag import Tag
from entities.post import Answer
from preprocessing import filters, parser, tags, preprocessing
from util.helper import APP_PATH


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


    def test_make_sure_tokens_that_are_tag_names_are_not_removed(self):
        # read in all tags
        all_tags, _, _ = parser.parse_tags_and_posts(os.path.join(APP_PATH, "data", "example"))

        filtered_tags, _ = tags.replace_tag_synonyms(all_tags, [])
        all_sorted_tag_names = sorted(map(lambda t: t.name, filtered_tags), reverse=True)
        text = ' '.join(all_sorted_tag_names)
        self.assertEqual(text.split(), all_sorted_tag_names)
        my_posts = [Post(1, text, text, set(filtered_tags[:2]), 10)]
        _, posts = main.preprocess_tags_and_posts(all_tags, my_posts, 0, enable_stemming=False,
                                                  replace_adjacent_tag_occurences=False)
        self.assertEqual(len(posts), 1)
        for idx, token in enumerate(posts[0].title_tokens):
            self.assertEqual(token, all_sorted_tag_names[idx])


    def test_make_sure_number_tokens_are_not_removed(self):
        text = 'Windows Server 2008 and then Web 2.0, but Apache 2.2 is more popular than IIS 7.0'
        tag1 = Tag('tag1', 1)
        my_posts = [Post(1, text, text, set([tag1]), 10)]
        _, posts = main.preprocess_tags_and_posts([tag1], my_posts, 0, enable_stemming=False,
                                                  replace_adjacent_tag_occurences=False)
        self.assertEqual(len(posts), 1)
        self.assertEqual(posts[0].title_tokens, ['windows', 'server', '2008', 'web', '2.0',
                                                 'apache', '2.2', 'popular', 'iis', '7.0'])


if __name__ == '__main__':
    unittest.main()
