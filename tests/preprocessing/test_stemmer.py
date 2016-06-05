import unittest
from entities.post import Post
from preprocessing import stemmer


class TestStemmer(unittest.TestCase):
    def test_porter_stemmer(self):
        post = Post(1, "", "", set([]), 1)

        post.title_tokens = ["writing"]
        post.body_tokens = ["house"]
        stemmer.porter_stemmer([post])
        self.assertEqual(["write"], post.title_tokens)
        self.assertEqual(["hous"], post.body_tokens)

        post.body_tokens = ["asp.net", "c++", "c#", "c", "So", "sixth", "abundantly", "had", "great", "yielding", "cattle", "together", "it", "him", "whales",
                       "rule", "air", "i", "lights", "yielding", "our", "green", "set", "forth", "years", "so",
                       "gathering", "land", "over"]
        expected_tokens = ["asp.net", "c++", "c#", "c", "So", "sixth", "abundantli", "had", "great", "yield", "cattl", "togeth", "it", "him", "whale",
                           "rule", "air", "i", "light", "yield", "our", "green", "set", "forth", "year", "so",
                           "gather", "land", "over"]

        stemmer.porter_stemmer([post])
        self.assertEqual(expected_tokens, post. body_tokens)

if __name__ == '__main__':
    unittest.main()
