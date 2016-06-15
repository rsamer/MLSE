# -*- coding: utf-8 -*-

import unittest
import os
import csv
import main
from entities.post import Post
from entities.tag import Tag
from entities.post import Answer
from preprocessing import filters, parser, tags, selection, stopwords, preprocessing as preproc # @UnresolvedImport
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
        self.assertEqual("this is a   test", post.body)

        post = Post(1, "title", "this is a<code>if else exit</code>test", set([]), 1)
        filters.strip_code_segments([post])
        self.assertEqual("this is a test", post.body)


    def test_parse_and_strip_code_segments(self):
        _, all_posts, _ = parser.parse_tags_and_posts(os.path.join(APP_PATH, "resources", "test"))

        self.assertEqual(len(all_posts), 3)
        selection.append_accepted_answer_text_to_body(all_posts)
        filters.strip_code_segments(all_posts)

        expected_body_text_after_stripping_code_segments = [
            '<p>A coworker of mine believes that <em>any</em> use of in-code comments (ie, not ' \
          + 'javadoc style method or class comments) is a <a href="http://en.wikipedia.org/wiki/' \
          + 'Code_smell">code smell</a>.  What do you think?</p>\n' \
          + ' <p>Ideally, code should be so well coded that it should be auto explicative. In ' \
          + 'the real world, we know that also very high quality code needs sometimes ' \
          + 'commenting.</p>\n\n<p>What you should absolutely avoid is "comment-code redundancy" ' \
          + '(comments that don\'t add anything to code):</p>\n\n<pre> </pre>\n\n<p>Then, if ' \
          + 'there\'s a good (and maintained/aligned) code design and documentation, commenting ' \
          + 'is even less useful.</p>\n\n<p>But in some circumstances comments can be a good aid ' \
          + 'in code readability:</p>\n\n<pre> </pre>\n\n<p>Don\'t forget that you have to ' \
          + 'maintain and keep in sync also comments... outdated or wrong comments can be a ' \
          + 'terrible pain! And, as a general rule, commenting too much can be a symptom of bad ' \
          + 'programming.</p>\n',

            '<p>Some relevant opinions that may be of interest:</p>\n\n<ul>\n' \
          + '<li><a href="http://www.python.org/dev/peps/pep-0008/">Guido says spaces</a></li>\n' \
          + '<li><a href="http://discuss.fogcreek.com/joelonsoftware/default.asp?cmd=show&amp;' \
          + 'ixPost=3978">Joel says spaces</a></li>\n<li><a href="http://www.codinghorror.com' \
          + '/blog/2009/04/death-to-the-space-infidels.html">Atwood says spaces</a></li>\n' \
          + '<li><a href="http://www.jwz.org/doc/tabs-vs-spaces.html">Zawinski says spaces, ' \
          + 'sort of</a></li>\n</ul>\n',

            '<p>Joel Spolsky wrote a famous blog post "<a href="http://www.joelonsoftware.com/' \
          + 'articles/fog0000000022.html" rel="nofollow">Human Task Switches considered harmful' \
          + '</a>".</p>\n\n<p>While I agree with the premise and it seems like common sense, ' \
          + 'I\'m wondering if there are any studies or white papers on this to calculate the ' \
          + 'overhead on task switches, or is the evidence merely anecdotal? </p>\n'
        ]
        for idx, post in enumerate(all_posts):
            self.assertEqual(len(post.body), len(expected_body_text_after_stripping_code_segments[idx]))
            self.assertEqual(post.body, expected_body_text_after_stripping_code_segments[idx])


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


    def test_parse_and_strip_code_segments_and_html_tags(self):
        _, all_posts, _ = parser.parse_tags_and_posts(os.path.join(APP_PATH, "resources", "test"))

        self.assertEqual(len(all_posts), 3)
        selection.append_accepted_answer_text_to_body(all_posts)
        filters.strip_code_segments(all_posts)
        filters.strip_html_tags(all_posts)

        expected_body_text_after_stripping_code_segments = [
            'A coworker of mine believes that any use of in-code comments (ie, not ' \
          + 'javadoc style method or class comments) is a code smell.  What do you think?\n' \
          + 'Ideally, code should be so well coded that it should be auto explicative. In ' \
          + 'the real world, we know that also very high quality code needs sometimes ' \
          + 'commenting.\nWhat you should absolutely avoid is "comment-code redundancy" ' \
          + '(comments that don\'t add anything to code):\n \nThen, if ' \
          + 'there\'s a good (and maintained/aligned) code design and documentation, commenting ' \
          + 'is even less useful.\nBut in some circumstances comments can be a good aid ' \
          + 'in code readability:\n \nDon\'t forget that you have to ' \
          + 'maintain and keep in sync also comments... outdated or wrong comments can be a ' \
          + 'terrible pain! And, as a general rule, commenting too much can be a symptom of bad ' \
          + 'programming.',

            'Some relevant opinions that may be of interest:\n\nGuido says spaces\nJoel says '
          + 'spaces\nAtwood says spaces\nZawinski says spaces, sort of',

            'Joel Spolsky wrote a famous blog post "Human Task Switches considered harmful' \
          + '".\nWhile I agree with the premise and it seems like common sense, ' \
          + 'I\'m wondering if there are any studies or white papers on this to calculate the ' \
          + 'overhead on task switches, or is the evidence merely anecdotal?'
        ]
        for idx, post in enumerate(all_posts):
            self.assertEqual(len(post.body), len(expected_body_text_after_stripping_code_segments[idx]))
            self.assertEqual(post.body, expected_body_text_after_stripping_code_segments[idx])


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

        post.body_tokens = ["_test_", "a", "bc", 'windows', 'xp']
        filters.filter_tokens([post], [])
        self.assertEqual(['a', 'bc', 'windows', 'xp'], post.body_tokens)

        post.title_tokens = ["a", "test"]
        post.body_tokens = ["a", "bc"]
        filters.filter_tokens([post], ["a"])
        self.assertEqual(["a", "test"], post.title_tokens)
        self.assertEqual(["a", "bc"], post.body_tokens)
        self.assertEqual(["a", "test", "a", "bc"], post.tokens(1))

        post.body_tokens = ["he's", "planets'", "you're", "we've", "isn't", "haven't"]
        filters.filter_tokens([post], [])
        self.assertEqual(["he", "planet", "you", "we"], post.body_tokens)

        post.body_tokens = ["#123", "123,45", "123.45", "1,234.56", "#FF0000", "#ff0000", "test"]
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
        all_tags, _, _ = parser.parse_tags_and_posts(os.path.join(APP_PATH, "resources", "test"))

        filtered_tags, _ = tags.replace_tag_synonyms(all_tags, [])
        all_sorted_tag_names = sorted(map(lambda t: t.name, filtered_tags), reverse=True)
        text = ' '.join(all_sorted_tag_names)
        self.assertEqual(text.split(), all_sorted_tag_names)
        my_posts = [Post(1, text, text, set(filtered_tags[:2]), 10)]
        _, posts = main.preprocess_tags_and_posts(all_tags, my_posts, 0, enable_stemming=False,
                                                  replace_adjacent_tag_occurences=False,
                                                  replace_token_synonyms_and_remove_adjacent_stopwords=False)
        self.assertEqual(len(posts), 1)
        idx_offset = 0
        for idx, token in enumerate(posts[0].title_tokens):
            if all_sorted_tag_names[idx + idx_offset] in stopwords.problematic_tag_names:
                idx_offset += 1
                continue
            self.assertEqual(token, all_sorted_tag_names[idx + idx_offset])


    def test_make_sure_number_tokens_are_not_removed(self):
        text = u'Windows Server 2008 costs 1,000,000 $ or 1,000,000.00$ or €1,000,000. Successor of ' \
             + u'Web 2.0 is Html5. In Apache 2.2 is more popular than IIS 7.0'
        tag1 = Tag('tag1', 1)
        my_posts = [Post(1, text, text, set([tag1]), 10)]
        _, posts = main.preprocess_tags_and_posts([tag1], my_posts, 0, enable_stemming=False,
                                                  replace_adjacent_tag_occurences=False,
                                                  replace_token_synonyms_and_remove_adjacent_stopwords=False)
        self.assertEqual(len(posts), 1)
        self.assertEqual(posts[0].title_tokens,
            ['windows', 'server', '2008', 'costs', '1,000,000', '$', '1,000,000.00', '$',
             '1,000,000', 'successor', 'web', '2.0', 'html', '5', 'apache', '2.2', 'popular',
             'iis', '7.0'])


    def test_full_preprocessing(self):
        all_tags, all_posts, _ = parser.parse_tags_and_posts(os.path.join(APP_PATH, "resources", "test"))
        _, posts = main.preprocess_tags_and_posts(all_tags, all_posts, 0, enable_stemming=False,
                                                  replace_adjacent_tag_occurences=False,
                                                  replace_token_synonyms_and_remove_adjacent_stopwords=False)
        self.assertEqual(len(all_posts), 3)
        self.assertEqual(len(posts), 3)

        expected_title_tokens = [
            ['comments', 'code', 'smell'],
            ['comments', 'code', 'smell'],
            ['data', 'human', 'task', 'switches', 'harmful']
        ]
        expected_body_tokens = [
            [
                u'coworker', u'mine', u'code', u'comments', u'javadoc', u'style', u'method',
                u'class', u'comments', u'code', u'smell', u'ideally', u'code', u'coded', u'auto',
                u'explicative', u'quality', u'code', u'comment', u'code', u'redundancy',
                u'comments', u'code', u'maintained', u'aligned', u'code', u'design',
                u'documentation', u'less', u'comments', u'code', u'readability',
                u'maintain', u'sync', u'comments', u'comments', u'rule', u'symptom',
                u'programming'
            ],
            [
                u'guido', u'spaces', u'joel', u'spaces', u'atwood', u'spaces', u'zawinski',
                u'spaces', u'sort'
            ],
            [
                u'joel', u'spolsky', u'wrote', u'famous', u'blog', u'post', u'human', u'task',
                u'switches', u'harmful', u'agree', u'premise', u'sense', u'studies', u'white',
                u'papers', u'calculate', u'overhead', u'task', u'switches', u'evidence',
                u'merely', u'anecdotal'
            ]
        ]
        for idx, post in enumerate(posts):
            self.assertEqual(post.title_tokens, expected_title_tokens[idx])
            self.assertEqual(post.body_tokens, expected_body_tokens[idx])


    def test_full_preprocessing_replacing_token_synonyms(self):
        all_tags, _, _ = parser.parse_tags_and_posts(os.path.join(APP_PATH, "resources", "test"))
        synonyms_file_path = os.path.join(APP_PATH, 'corpora', 'tokens', 'synonyms')
        all_source_token_parts = []
        all_target_token_parts = []
        with open(synonyms_file_path, 'rb') as f:
            reader = csv.reader(f)
            for row in reader:
                all_source_token_parts.append(row[1].strip())
                all_target_token_parts.append(row[0].strip())
        self.assertEqual(len(all_source_token_parts), len(all_target_token_parts))

        # test single synonyms
        for idx, source_token in enumerate(all_source_token_parts):
            post = Post(1, source_token, source_token, set(all_tags[:3]), 1)
            _, [post] = main.preprocess_tags_and_posts(all_tags, [post], 0, enable_stemming=False,
                                                       replace_adjacent_tag_occurences=False,
                                                       replace_token_synonyms_and_remove_adjacent_stopwords=True)
            self.assertEqual(post.title_tokens, all_target_token_parts[idx].split())
            self.assertEqual(post.body_tokens, all_target_token_parts[idx].split())

        # test all synonyms in one string
        post = Post(1, ' '.join(all_source_token_parts), ' '.join(all_source_token_parts), set(all_tags[:3]), 1)
        _, [post] = main.preprocess_tags_and_posts(all_tags, [post], 0, enable_stemming=False,
                                                   replace_adjacent_tag_occurences=False,
                                                   replace_token_synonyms_and_remove_adjacent_stopwords=True)
        self.assertEqual(' '.join(post.title_tokens), ' '.join(all_target_token_parts).strip())
        self.assertEqual(' '.join(post.body_tokens), ' '.join(all_target_token_parts).strip())


if __name__ == '__main__':
    unittest.main()
