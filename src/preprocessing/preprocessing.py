
import logging
from entities.post import Post
from entities.tag import Tag

log = logging.getLogger("preprocessing.preprocessing")

def preprocess_tags_and_sort_by_frequency(tags, tag_frequency_threshold):
    assert(isinstance(tags, list))
    sorted_tags = sorted(tags, key=lambda x:x[1].count, reverse=True)
    filtered_tags = [tag for (_, tag) in sorted_tags if tag.count >= tag_frequency_threshold]
    return filtered_tags

def preprocess_posts(posts):
    assert(isinstance(posts, list))

    #-------
    import re
    regex_str = [
        r'C\+\+',
        r'C\#',
        r'\w+\.\w+',
        r'<[^>]+>', # HTML tags
        r'(?:@[\w_]+)', # @-mentions
        r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", # hash-tags
        r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', # URLs
    
        r'(?:(?:\d+,?)+(?:\.?\d+)?)', # numbers
        r"(?:[a-z][a-z'\-_]+[a-z])", # words with - and '
        r'(?:[\w_]+)', # other words
        r'(?:\S)' # anything else
    ]
    tokens_re = re.compile(r'('+'|'.join(regex_str)+')', re.VERBOSE | re.IGNORECASE)

    def strip_html_tags(posts):
        try:
            from bs4 import BeautifulSoup #@UnresolvedImport
        except ImportError:
            raise RuntimeError('Please install BeautifulSoup library!')
    
        for post in posts:
            assert(isinstance(post, Post))
            post.body = BeautifulSoup(post.body, "html.parser").text.strip()
        return posts

    def tokenize_posts(posts):
        def tokenize(s):
            return tokens_re.findall(s)

        def preprocess(s, lowercase=False):
            tokens = tokenize(s)
            if lowercase:
                tokens = [token.lower() for token in tokens]
            return tokens

        for post in posts:
            post.body_tokens = []
            # TODO: ...
            print(post.body)
            #print(preprocess(post.body))
            
            tweet = "RT @marcobonzanini: just an example! :D http://example.com #NLP"
            print(preprocess(tweet))
            # ['RT', '@marcobonzanini', ':', 'just', 'an', 'example', '!', ':D', 'http://example.com', '#NLP']
            
            print(preprocess("C++ is a :=; := :D complicated programming-language compared to C#. Asp.Net"))

            #for word in post.body.split():
            #    pass
                
        return posts

    posts = strip_html_tags(posts)
    posts = tokenize_posts(posts)
    return posts
