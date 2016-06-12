import unittest
from entities.post import Post
from entities.tag import Tag
from preprocessing import tags


class TestTags(unittest.TestCase):
    def test_replace_adjacent_tag_occurences(self):
        post = Post(1, "", "", set([]), 1)

        post.title = "test ing design patterns object oriented-design."
        post.body = "programming languages test and so forth"
        tag_names = ["programming-language", "design-patterns", "object-oriented"]
        tags.replace_adjacent_tag_occurences([post], tag_names)
        from preprocessing import tokenizer
        tokenizer.tokenize_posts([post], tag_names)
        self.assertEqual("test ing  design-patterns   object-oriented -design.", post.title)
        self.assertEqual(" programming-language s test and so forth", post.body)

    def test_strip_invalid_tags_from_posts_and_remove_untagged_posts(self):
        post1 = Post(1, "", "", set([]), 1)
        post2 = Post(2, "", "", set([]), 1)

        tag1 = Tag("tag1", 1)
        tag2 = Tag("tag2", 1)
        tag3 = Tag("tag3", 1)
        tag4 = Tag("tag4", 1)
        tag_test = Tag("tag-test", 1)

        post1.tag_set = {tag1}
        post2.tag_set = {tag_test}
        posts = [post1, post2]
        posts = tags.strip_invalid_tags_from_posts_and_remove_untagged_posts(posts, [tag_test])
        self.assertEqual([post2], posts)
        self.assertEqual(1, len(post2.tag_set))

        post1.tag_set = {tag1, tag2}
        post2.tag_set = {tag3, tag4}
        posts = [post1, post2]
        posts = tags.strip_invalid_tags_from_posts_and_remove_untagged_posts(posts, [tag1, tag4])
        self.assertEqual([post1, post2], posts)
        self.assertEqual(1, len(post1.tag_set))
        self.assertEqual(tag1, list(post1.tag_set)[0])
        self.assertEqual(tag4, list(post2.tag_set)[0])

    def test_replace_tag_synonyms(self):
        post = Post(1, "title", "body", set([]), 1)
        tag1 = Tag("tag1", 1)
        tag2 = Tag("apache", 1)
        tag3 = Tag("apache2", 1)

        post.tag_set = {tag1, tag2, tag3}

        tag_list = [tag1, tag2, tag3]
        self.assertEqual(3, len(tag_list))

        post.tag_set = tag_list
        tag_list, post_list = tags.replace_tag_synonyms(tag_list, [post])

        self.assertEqual(2, len(tag_list))
        self.assertEqual(tag1, tag_list[0])
        self.assertEqual(tag2, tag_list[1])

        self.assertEqual(2, len(post.tag_set))
        self.assertNotEquals(list(post.tag_set)[0], list(post.tag_set)[1])
        self.assertTrue(list(post.tag_set)[0] == tag1 or list(post.tag_set)[0] == tag2)
        self.assertTrue(list(post.tag_set)[1] == tag1 or list(post.tag_set)[1] == tag2)


if __name__ == '__main__':
    unittest.main()
