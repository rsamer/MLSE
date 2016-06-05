import unittest
from entities.post import Post
from entities.post import Answer
from preprocessing import selection


class TestSelection(unittest.TestCase):
    def test_add_title_to_body(self):
        post = Post(1, "", "", set([]), 1)

        post.title_tokens = ["title", "test"]
        post.body_tokens = ["body", "test"]
        self.assertEqual(["title", "test"] * 10 + ["body", "test"], post.tokens(10))

        post.title_tokens = ["title"]
        post.body_tokens = ["body"]
        self.assertEqual(["body"], post.tokens(0))

    def test_add_accepted_answer_text_to_body(self):
        post = Post(1, "title", "body", set([]), 1)

        accepted_answer = Answer(2, "answer", 1)
        post.accepted_answer_id = accepted_answer.pid
        post.answers = [accepted_answer]

        selection.add_accepted_answer_text_to_body([post])
        self.assertEqual("title", post.title)
        self.assertEqual("body answer", post.body)

        post.body = "body"
        accepted_answer.score = 0
        selection.add_accepted_answer_text_to_body([post])
        self.assertEqual("body answer", post.body)

        post.body = "body"
        accepted_answer.score = -1
        selection.add_accepted_answer_text_to_body([post])
        self.assertEqual("body", post.body)

        post.body = "body"
        post.accepted_answer_id = None
        selection.add_accepted_answer_text_to_body([post])
        self.assertEqual("body", post.body)

        post.body = "body"
        post.accepted_answer_id = 1000
        selection.add_accepted_answer_text_to_body([post])
        self.assertEqual("body", post.body)

if __name__ == '__main__':
    unittest.main()
