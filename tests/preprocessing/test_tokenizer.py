# -*- coding: utf-8 -*-
import unittest
from entities.post import Post
from preprocessing import tokenizer


class TestTokenizer(unittest.TestCase):
    def test_tokenizer(self):

        # own examples:
        self.assert_tokens(u"RT @marcobonzanini: just, an example! http://example.com/what?q=test #NLP", [],
                           ["rt", "@marcobonzanini", "just", "an", "example", "#nlp"])
        self.assert_tokens(u"0x2AF3 #143152 A b C d e f g h i j k f# u# and C++ is a test hehe ", ["c++", "f#"],
                           ["0", "x2af3", "#143152", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k",
                            "f#", "u", "#", "and", "c++", "is", "a", "test", "hehe"])
        self.assert_tokens(u"wt iop wt-iop wt*iop ip address complicated programming-languages object oriented", [],
                           ["wt", "iop", "wt-iop", "wt", "*", "iop", "ip", "address", "complicated", "programming-languages", "object", "oriented"])
        self.assert_tokens(u"object-oriented-design compared to C#. AT&T Asp.Net C++!!", ["c#", "c++"],
                           ["object-oriented-design", "compared", "to", "c#", "at&t", "asp.net", "c++"])
        self.assert_tokens(u"C++~$§%) is a :=; := :D :-)) ;-)))) programming language!", ["C++"],
                           ["c++", "$", u"§", "%", "is", "a", "=", "=", "programming", "language"])
        self.assert_tokens(u"Blue houses are... ~ hehe wt~iop complicated programming-language compared to C#. AT&T Asp.Net C++ #1234 1234!!", ["C#", "C++"],
                           ["blue", "houses", "are", "hehe", "wt", "iop", "complicated", "programming-language", "compared", "to", "c#",
                            "at&t", "asp.net", "c++", "#1234", "1234"])

        # check if simple smilies are automatically removed by tokenizer:
        # Note: the more complexer ones will be removed later on by the filter module (See: filters.py)
        self.assert_tokens("Smilies to be removed: ;-) :-D :-P", [], ["smilies", "to", "be", "removed"])
        self.assert_tokens("Smilies to be removed: ;) :D :P", [], ["smilies", "to", "be", "removed"])

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

#     Concurrency issues are very hard to debug and very important to find because nowadays most devices have a multicore cpu.
#     Choosing a Java Web Framework now?
#     we are in the planning stage of migrating a large website which is built on a custom developed
#     mvc framework to a java based web framework which provides built-in support for ajax, rich media content,
#     mashup, templates based layout, validation, maximum html/java code separation.
#     Grails looked like a good choice, however, we do not want to use a scripting language.
#     We want to continue using java. Template based layout is a primary concern as we intend to use
#     this web application with multiple web sites with similar functionality but radically different
#     look and feel. Is portal based solution a good fit to this problem? Any insights on using
#     "Spring Roo" or "Play" will be very helpful. I did find similar posts like this, but it is more than a year old.
#     Things have surely changed in the mean time! EDIT 1: Thanks for the great answers!
#     This site is turning to be the best single source for in-the-trenches programmer info.
#     However, I was expecting more info on using a portal-cms duo.
#     Jahia looks goods. Anything similar?

        self.assert_tokens("this is a test", [], ["this", "is", "a", "test"])
        self.assert_tokens("this, is a test", [], ["this", "is", "a", "test"])
        self.assert_tokens("ASP.net is a programming-language", [], ["asp.net", "is", "a", "programming-language"])
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

    def assert_tokens(self, body, tag_names, expected_tokens):
        post = Post(1, "title", body, set([]), 1)

        tokenizer.tokenize_posts([post], tag_names=tag_names)
        self.assertEqual(expected_tokens, post.tokens)


if __name__ == '__main__':
    unittest.main()
