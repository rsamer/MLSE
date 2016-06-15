# -*- coding: utf-8 -*-
import unittest
from entities.post import Post
from entities.post import Answer
from preprocessing import tokenizer, preprocessing as prepr # @UnresolvedImport


class TestAppendAnswer(unittest.TestCase):
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

        prepr.append_accepted_answer_text_to_body([post])
        self.assertEqual("title", post.title)
        self.assertEqual("body answer", post.body)

        post.body = "body"
        accepted_answer.score = 0
        prepr.append_accepted_answer_text_to_body([post])
        self.assertEqual("body answer", post.body)

        post.body = "body"
        accepted_answer.score = -1
        prepr.append_accepted_answer_text_to_body([post])
        self.assertEqual("body", post.body)

        post.body = "body"
        post.accepted_answer_id = None
        prepr.append_accepted_answer_text_to_body([post])
        self.assertEqual("body", post.body)

        post.body = "body"
        post.accepted_answer_id = 1000
        prepr.append_accepted_answer_text_to_body([post])
        self.assertEqual("body", post.body)


class TestTokenizer(unittest.TestCase):
    def test_tokenizer_simple_testcases(self):
        self.assert_tokens("this is a test", [], ["this", "is", "a", "test"])
        self.assert_tokens("this, is a test", [], ["this", "is", "a", "test"])
        self.assert_tokens("ASP.net is not a programming-language", [],
                           ["asp.net", "is", "not", "a", "programming-language"])
        self.assert_tokens("C++", [], ["c", "+", "+"]) # split special characters when this word is NO tag!
        self.assert_tokens("C++", ["c++"], ["c++"]) # do not split special characters when this word is a tag!
        self.assert_tokens("C++", ["C++"], ["c++"]) # do not split special characters when this word is a tag!
        self.assert_tokens("C++, is a programming language", ["C++"], ["c++", "is", "a", "programming", "language"])
        self.assert_tokens("first.second", [], ["first.second"])
        self.assert_tokens("first. second", [], ["first", "second"])
        self.assert_tokens("tag-Name", [], ["tag-name"])
        self.assert_tokens("tag-Name!", [], ["tag-name"])
        self.assert_tokens("this ! is a test", [], ["this", "is", "a", "test"])
        self.assert_tokens("this . is a test", [], ["this", "is", "a", "test"])
        self.assert_tokens("this , is a test", [], ["this", "is", "a", "test"])
        self.assert_tokens("this ) is a test", [], ["this", "is", "a", "test"])
        self.assert_tokens("(this) is a test", [], ["this", "is", "a", "test"])
        self.assert_tokens("this: is a test", [], ["this", "is", "a", "test"])
        self.assert_tokens("this - is a test", [], ["this", "is", "a", "test"])
        self.assert_tokens("this? is a test", [], ["this", "is", "a", "test"])
        self.assert_tokens("don't", [], ["don't"])
        self.assert_tokens("do not", [], ["do", "not"])
        self.assert_tokens("hello 1234", [], ["hello", "1234"])
        #prepr.important_words_for_tokenization(tag_names)


    def test_tokenizer_complex_testcase(self):
        # known tags: []
        self.assert_tokens('html5 is the latest version of html', [],
                           ['html', '5', 'is', 'the', 'latest', 'version', 'of', 'html'])

        # check if "html5" is recognized as own term when "html5" is a known tag
        # known tags: ['html5']
        self.assert_tokens('html5 is the successor of html', ['html5'],
                           ['html5', 'is', 'the', 'successor', 'of', 'html'])

        # known tags: ['c#', 'c++']
        self.assert_tokens(u"object-oriented-design compared to C#. AT&T Asp.Net C++!!", ["c#", "c++"],
                           ["object-oriented-design", "compared", "to", "c#", "at&t",
                            "asp.net", "c++"])

        # known tags: []
        self.assert_tokens(u"RT @marcobonzanini: just, an example! http://example.com/what?q=test #NLP", [],
                           ["rt", "@marcobonzanini", "just", "an", "example", "#nlp"])

        # known tags: ['c++', 'f#']
        self.assert_tokens(u"0x2AF3 #143152 A b C d e f g h i j k f# u# and C++ is a test hehe ", ["c++", "f#"],
                           ["0", "x2af3", "#143152", "a", "b", "c", "d", "e", "f", "g", "h", "i",
                            "j", "k", "f#", "u", "#", "and", "c++", "is", "a", "test", "hehe"])

        # known tags: []
        self.assert_tokens(u"wt iop wt-iop wt*iop ip address complicated programming-languages object oriented", [],
                           ["wt", "iop", "wt-iop", "wt", "*", "iop", "ip", "address", "complicated",
                            "programming-languages", "object", "oriented"])

        # known tags: ['C++']
        self.assert_tokens(u"C++~$§%) is a :=; := :D :-)) ;-)))) programming language!", ["C++"],
                           ["c++", "$", "%", "is", "a", "=", "=", "programming", "language"])

        # known tags: ['C#', 'C++']
        self.assert_tokens(u"Blue houses are... ~ hehe wt~iop complicated programming-language compared to C#. AT&T Asp.Net C++ #1234 1234!!", ["C#", "C++"],
                           ["blue", "houses", "are", "hehe", "wt", "iop", "complicated",
                            "programming-language", "compared", "to", "c#", "at&t", "asp.net",
                            "c++", "#1234", "1234"])


    def test_tokenizer_smilies_within_text(self):
        # check if simple smilies are automatically removed by tokenizer:
        # Note: the more complexer ones will be removed later on by the filter module (See: filters.py)
        self.assert_tokens("Smilies to be removed: ;-) :-D :-P", [], ["smilies", "to", "be", "removed"])
        self.assert_tokens("Smilies to be removed: ;) :D :P", [], ["smilies", "to", "be", "removed"])


    def test_tokenizer_real_world_examples_from_dataset(self):
        # from real examples:
        self.assert_tokens("Do dynamic typed languages deserve all the criticism?", [],
                           ["do", "dynamic", "typed", "languages", "deserve", "all", "the", "criticism"])
        self.assert_tokens("I have read a few articles on the Internet about programming language choice in the enterprise.", [],
                           ["i", "have", "read", "a", "few", "articles", "on", "the", "internet",
                            "about", "programming", "language", "choice", "in", "the", "enterprise"])
        self.assert_tokens("Recently many dynamic typed languages have been popular, i.e. Ruby, Python, PHP and Erlang.", [],
                           ["recently", "many", "dynamic", "typed", "languages", "have", "been", "popular", "i.e",
                            "ruby", "python", "php", "and", "erlang"])
        self.assert_tokens("But many enterprises still stay with static typed languages like C, C++, C# and Java.", ["C++", "C#", "Java"],
                           ["but", "many", "enterprises", "still", "stay", "with", "static", "typed", "languages",
                            "like", "c", "c++", "c#", "and", "java"])
        self.assert_tokens("And yes, one of the benefits of static typed languages is that programming errors are caught earlier, at compile time, rather than at run time.", [],
                           ["and", "yes", "one", "of", "the", "benefits", "of", "static", "typed",
                            "languages", "is", "that", "programming", "errors", "are", "caught",
                            "earlier", "at", "compile", "time", "rather", "than", "at", "run", "time"])
        self.assert_tokens("But there are also advantages with dynamic typed languages. (more on Wikipedia)", [],
                           ["but", "there", "are", "also", "advantages", "with", "dynamic", "typed", "languages",
                            "more", "on", "wikipedia"])
        self.assert_tokens("The main reason why enterprises don't start to use languages like Erlang, Ruby and Python, seem to be the fact that they are dynamic typed.", [],
                           ["the", "main", "reason", "why", "enterprises", "don't", "start", "to", "use",
                            "languages", "like", "erlang", "ruby", "and", "python", "seem", "to", "be",
                            "the", "fact", "that", "they", "are", "dynamic", "typed"])
        self.assert_tokens("That also seem to be the main reason why people on StackOverflow decide against Erlang.", [],
                           ["that", "also", "seem", "to", "be", "the", "main", "reason", "why",
                            "people", "on", "stackoverflow", "decide", "against", "erlang"])
        self.assert_tokens("See Why did you decide against Erlang. However, there seem to be a strong criticism against "
                           + "dynamic typing in the enterprises, but I don't really get it why it is that strong.", [],
                           ["see", "why", "did", "you", "decide", "against", "erlang", "however", "there",
                            "seem", "to", "be", "a", "strong", "criticism", "against", "dynamic", "typing",
                            "in", "the", "enterprises", "but", "i", "don't", "really", "get", "it",
                            "why", "it", "is", "that", "strong"])
        self.assert_tokens("Really, why is there so much criticism against dynamic typing in the enterprises?", [],
                           ["really", "why", "is", "there", "so", "much", "criticism", "against", "dynamic",
                            "typing", "in", "the", "enterprises"])
        self.assert_tokens("Does it really affect the cost of projects that much, or what? But maybe I'm wrong.", [],
                           ["does", "it", "really", "affect", "the", "cost", "of", "projects", "that",
                            "much", "or", "what", "but", "maybe", "i'm", "wrong"])
        self.assert_tokens("Java.util.List thread-safe? Is a java.util.List thread-safe?", [],
                           ["java.util.list", "thread-safe", "is", "a", "java.util.list", "thread-safe"])
        self.assert_tokens("From C++ I know that std::vectors are not thread-safe.d.f", ["C++"],
                           ['from', 'c++', 'i', 'know', 'that', 'std', 'vectors', 'are', 'not', 'thread-safe.d.f'])
        self.assert_tokens("Concurrency issues are very hard to debug and very important to find because nowadays most devices have a multicore cpu.", ["C++"],
                           ['concurrency', 'issues', 'are', 'very', 'hard', 'to', 'debug', 'and',
                            'very', 'important', 'to', 'find', 'because', 'nowadays', 'most',
                            'devices', 'have', 'a', 'multicore', 'cpu'])
        self.assert_tokens("Choosing a Java Web Framework now?", ["java"],
                           ['choosing', 'a', 'java', 'web', 'framework', 'now'])
        self.assert_tokens("we are in the planning stage of migrating a large website which is built on a custom developed", [],
                           ['we', 'are', 'in', 'the', 'planning', 'stage', 'of', 'migrating', 'a',
                            'large', 'website', 'which', 'is', 'built', 'on', 'a', 'custom',
                            'developed'])
        self.assert_tokens("mvc framework to a java based web framework which provides built-in " + \
                           "support for ajax, rich media content,", [],
                           ['mvc', 'framework', 'to', 'a', 'java', 'based', 'web', 'framework',
                            'which', 'provides', 'built-in', 'support', 'for', 'ajax', 'rich',
                            'media', 'content'])
        self.assert_tokens('"Spring Roo" or "Play" will be very helpful.', [],
                           ['spring', 'roo', 'or', 'play', 'will', 'be', 'very', 'helpful'])
        self.assert_tokens('Grails looked like a good choice, however, EDIT 1: Thanks!', [],
                           ['grails', 'looked', 'like', 'a', 'good', 'choice', 'however', 'edit',
                            '1', 'thanks'])
        self.assert_tokens('mashup, templates based layout, validation, maximum html/java code separation.', [],
                           ['mashup', 'templates', 'based', 'layout', 'validation', 'maximum',
                            'html', 'java', 'code', 'separation'])


    def assert_tokens(self, body, important_words, expected_tokens):
        post = Post(1, "title ! - :-)", body, set([]), 1)
        tokenizer.tokenize_posts([post], important_words)
        self.assertEqual(["title"], post.title_tokens)
        self.assertEqual(expected_tokens, post.body_tokens)


if __name__ == '__main__':
    unittest.main()
