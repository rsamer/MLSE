
import re
import logging
from entities.post import Post
from entities.tag import Tag

log = logging.getLogger("preprocessing.preprocessing")

tokens_punctuation_re = re.compile(r"(\.|!|\?|\(|\))$")
single_character_tokens_re = re.compile(r"^\W$")

def preprocess_tags_and_sort_by_frequency(tags, frequency_threshold):
    assert isinstance(tags, list)
    sorted_tags = sorted(tags, key=lambda x: x[1].count, reverse=True)
    return [tag for (_, tag) in sorted_tags if tag.count >= frequency_threshold]

def preprocess_posts(posts, tags):
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
    
            #print "-"*80
            #print post.body + "\n"
            #-------------------
            # FIXME: remove source-code from body before!!!
            #        Figure out how this is best realized with BeautifulSoup...
            #-------------------
            #print(post.body_tokens)
            # TODO: ...
            #for word in post.body.split():
            #    pass
            #print "-"*80
            post.body_tokens = tokens

    def _strip_html_tags(posts):
        try:
            from bs4 import BeautifulSoup #@UnresolvedImport
        except ImportError:
            raise RuntimeError('Please install BeautifulSoup library!')
    
        for post in posts:
            assert(isinstance(post, Post))
            post.body = BeautifulSoup(post.body, "html.parser").text.strip()

    def _tokenize_posts(posts, tag_names):
        ''' Customized tokenizer for our special needs! (e.g. C#, C++, ...) '''
        # based on: http://stackoverflow.com/a/36463112
        regex_str = [
            r'\w+\.\w+',
            r'(?:@[\w_]+)', # @-mentions
            r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", # hash-tags
            r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', # URLs
            #r'(?:\d+\%)', # percentage
            r'(?:(?:\d+,?)+(?:\.?\d+)?)', # numbers
            r"(?:[a-z][a-z'\-_]+[a-z])", # words with - and '
        # avoid splitting adjacent words that are known tag names
        ] + map(lambda tag_name: re.escape(tag_name) + '\W', tag_names) + [
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
        print(_tokenize_text("RT @marcobonzanini: just an example! :D http://example.com #NLP", tag_names))
        print(_tokenize_text("C++ is a :=; := :D complicated programming-language compared to C#. Asp.Net C++ ", tag_names))

    assert isinstance(posts, list)
    tag_names = [tag.name.lower() for tag in tags]

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

    _strip_html_tags(posts)
    _tokenize_posts(posts, tag_names)
    _filter_tokens(posts, tag_names)

    #import sys; sys.exit()

    # TODO: remove numbers starting with "#" (e.g. #329410)
    # TODO: remove hex-numbers
    # TODO: remove very unique words that only occur once in the whole dataset!!

    return posts
