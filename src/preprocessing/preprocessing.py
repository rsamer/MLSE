# -*- coding: utf-8 -*-

import re
import logging
from entities.post import Post
from entities.tag import Tag

log = logging.getLogger("preprocessing.preprocessing")

tokens_punctuation_re = re.compile(r"(\.|!|\?|\(|\)|~)$")
single_character_tokens_re = re.compile(r"^\W$")

def preprocess_tags_and_sort_by_frequency(tags, frequency_threshold):
    assert isinstance(tags, list)
    sorted_tags = sorted(tags, key=lambda x: x[1].count, reverse=True)
    # TODO: look for similar tag names and merge them together, e.g. "java", "java programming"??
    #       for this we should simply download the synonym list from StackExchange as mentioned in the paper!
    #
    # TODO: also remove all removed tags from posts in order to avoid data-inconsistency issues!!!
    return [tag for (_, tag) in sorted_tags if tag.count >= frequency_threshold]

def preprocess_posts(posts, tags):

    def _strip_html_tags(posts):
        try:
            from bs4 import BeautifulSoup #@UnresolvedImport
        except ImportError:
            raise RuntimeError('Please install BeautifulSoup library!')
    
        for post in posts:
            assert(isinstance(post, Post))
            #-------------------
            # TODO:  Problem: source-code inside the body is interpreted as normal text...
            #        => Remove source-code from body before!!!
            #        => Figure out how this is best realized with BeautifulSoup...
            #-------------------
            post.body = BeautifulSoup(post.body, "html.parser").text.strip()

    def _tokenize_posts(posts, tag_names):
        ''' Customized tokenizer for our special needs! (e.g. C#, C++, ...) '''
        # based on: http://stackoverflow.com/a/36463112
        regex_str = [
            r'(?:[:;=\^\-oO][\-_\.]?[\)\(\]\[\-DPOp_\^\\\/])', # emoticons (Note: they are removed after tokenization!)
            r'\w+\.\w+',
            r'\w+\&\w+',    # e.g. AT&T
            r'(?:@[\w_]+)', # @-mentions
            r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", # hash-tags
            r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', # URLs  (Note: they are removed after tokenization!)
            #r'(?:\d+\%)', # percentage
            r'(?:(?:\d+,?)+(?:\.?\d+)?)', # numbers
            r"(?:[a-z][a-z'\-_]+[a-z])", # words with - and '
        ] + map(lambda tag_name: re.escape(tag_name) + '\W', tag_names) + [ # consider known tag names
            r'(?:[\w_]+)', # other words
            r'(?:\S)' # anything else
        ]
        tokens_ignore_re = re.compile(r'('+'|'.join(regex_str)+')', re.VERBOSE | re.IGNORECASE)

        def _tokenize_text(s, tag_names, lowercase=True):
            def tokenize(s):
                return tokens_ignore_re.findall(s)
    
            def filter_all_tailing_punctuation_characters(token):
                while True:
                    punctuation_character = tokens_punctuation_re.findall(token)
                    if len(punctuation_character) == 0:
                        return token
                    token = token[:-1]

            # prepending and appending a single whitespace to the text
            # makes the regular expressions less complicated
            tokens = tokenize(" " + s + " ")
            if lowercase:
                tokens = [token.lower() for token in tokens]
            tokens = [token.strip() for token in tokens] # remove whitespaces before and after
            tokens = map(filter_all_tailing_punctuation_characters, tokens)
            return tokens

        for post in posts:
            post.body_tokens = _tokenize_text(post.body, tag_names)

    def _filter_tokens(posts, tag_names):
        for post in posts:
            tokens = post.body_tokens
            # remove empty tokens, numbers and those single-character words that are no letters
            tokens = filter(lambda t: len(t) > 0 and not t.isdigit() and len(single_character_tokens_re.findall(t)) == 0, tokens)
    
            # remove single-character words that are not part of our tag list
            tokens = filter(lambda t: len(t) > 1 or t in tag_names, tokens)
    
            # remove "'s" at the end
            # "'s" is not useful for us because it may only indicate the verb "be", "have"
            # (e.g. "He's got ..." or "It's ...")
            # or may indicate a possessive nouns (e.g. His wife's shoes...)
            tokens = map(lambda t: t[:-2] if t.endswith("'s") else t, tokens)
    
            # remove "s'" at the end
            # "s'" is also not useful for us since it indicates the plural version
            # of possessive nouns (e.g. the planets' orbits)
            tokens = map(lambda t: t[:-2] if t.endswith("s'") else t, tokens)

            # remove "'ve" at the end (e.g. we've got...)
            tokens = map(lambda t: t[:-2] if t.endswith("'ve") else t, tokens)

            post.body_tokens = tokens


    assert isinstance(posts, list)
    tag_names = [tag.name.lower() for tag in tags]

    # DEBUG BEGIN
    test_post1 = Post(1, "", u"RT @marcobonzanini: just, an example! :D http://example.com/what?q=test #NLP", [])
    test_post2 = Post(2, "", u"C++ is a test hehe wt iop complicated programming-language object oriented object-oriented-design compared to C#. AT&T Asp.Net C++!!", [])
    test_post3 = Post(3, "", u"C++~$§%) is a :=; := :D test ~ hehe wt~iop complicated programming-language compared to C#. AT&T Asp.Net C++!!", [])
    posts += [test_post1, test_post2, test_post3]
    # DEBUG END

    _strip_html_tags(posts)
    _tokenize_posts(posts, tag_names)
    _filter_tokens(posts, tag_names)
    # TODO: remove emoticons and URLs in _filter_tokens (right AFTER tokenization and NOT before!!)
    # TODO: remove numbers starting with "#" (e.g. #329410) in _filter_tokens
    # TODO: remove hex-numbers in _filter_tokens
    # TODO: remove very unique words that only occur once in the whole dataset _filter_tokens ??!!

    # DEBUG BEGIN
    print "\n" + ("-"*80)
    print test_post1.body_tokens
    print test_post2.body_tokens
    print test_post3.body_tokens
    print "-"*80
    # DEBUG END

    #--------------------------------------------------------------------------------
    # TODO: replace "-" by " " in all tag names e.g. "object-oriented" -> "object oriented"
    #       and then look for two (or more) adjacent words that represent a known tag name
    #
    #       e.g. current token list ["I", "love", "object", "oriented", "code"]
    #     -> should be converted to ["I", "love", "object-oriented", "code"]
    #        since "object-oriented" is a tag name in our tag list
    #--------------------------------------------------------------------------------
    return posts




#---------------------------------------------------------------------------------------
#
# TODO: use python's multiprocess modules,
#       since this multithreaded solution is not efficient... (see: https://wiki.python.org/moin/GlobalInterpreterLock CPython's GIL)
#
#---------------------------------------------------------------------------------------
#
#     # multithreaded
#     from threading import Thread
#     class _PreprocessingThread(Thread):
#         def run(self):
#             kwargs = self._Thread__kwargs
#             my_posts = kwargs["posts"]
#             tag_names = kwargs["tag_names"]
#             _strip_html_tags(my_posts)
#             _tokenize_posts(my_posts, tag_names)
#             _filter_tokens(my_posts, tag_names)
#
#     NUMBER_OF_THREADS = 10
#     number_of_posts_per_thread = len(posts) / NUMBER_OF_THREADS
#     total_number_of_posts = len(posts)
#     all_threads = []
#     new_start_index = 0
#     for _ in range(NUMBER_OF_THREADS):
#         start_index = new_start_index
#         end_index = min(start_index + number_of_posts_per_thread, (total_number_of_posts - 1))
#         posts_for_thread = posts[start_index:(end_index+1)]
#         thread = _PreprocessingThread(kwargs={ "posts": posts_for_thread, "tag_names": tag_names })
#         all_threads.append(thread)
#         new_start_index = end_index + 1
#
#     for thread in all_threads:
#         thread.start()
#     for thread in all_threads:
#         thread.join()
