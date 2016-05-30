import unittest
from entities.post import Post
from preprocessing import tags


class TestTags(unittest.TestCase):
    def test_replace_adjacent_tag_occurences(self):
        post = Post(1, "", "", set([]), 1)

        post.title = "test ing design patterns ."
        post.body = "programming languages test and so forth"
        tags.replace_adjacent_tag_occurences([post], ["programming-language", "design-patterns"])
        self.assertEqual("test ing design-patterns .", post.title)
        self.assertEqual("programming-languages test and so forth", post.body)

    def test_strip_invalid_tags_from_posts_and_remove_untagged_posts(self):
        post1 = Post(1, "", "", set([]), 1)
        post2 = Post(2, "", "", set([]), 1)

        post1.tag_set = {"tag1"}
        post2.tag_set = {"test"}
        posts = [post1, post2]
        posts = tags.strip_invalid_tags_from_posts_and_remove_untagged_posts(posts, ["test"])
        self.assertEqual([post2], posts)
        self.assertEqual(1, len(post2.tag_set))

        post1.tag_set = {"tag1", "tag2"}
        post2.tag_set = {"tag3", "tag4"}
        posts = [post1, post2]
        posts = tags.strip_invalid_tags_from_posts_and_remove_untagged_posts(posts, ["tag1", "tag4"])
        self.assertEqual([post1, post2], posts)
        self.assertEqual(1, len(post1.tag_set))
        self.assertEqual("tag1", list(post1.tag_set)[0])
        self.assertEqual("tag4", list(post2.tag_set)[0])

if __name__ == '__main__':
    unittest.main()
