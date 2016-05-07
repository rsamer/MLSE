#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
    This script generates word cloud of top 100 tags from the given StackExchange data set

    Note: You may have to install the following libraries first:
    * wordcloud (pip install wordcloud)
    * matplotlib

    usage:
    python -m generate_tag_cloud ../data/example/

    -----------------------------------------------------------------------------
    NOTE: the data set located in ../data/example/ is NOT a representative subset
          of the entire "programmers.stackexchange.com" data set
    -----------------------------------------------------------------------------
'''

import sys
import os

from main import usage
from entities.tag import Tag

try:
    from wordcloud import WordCloud
except ImportError:
    raise RuntimeError('Please install wordcloud library!')

try:
    import matplotlib.pyplot as plt
except ImportError:
    raise RuntimeError('Please install matplotlib library!')

def main():
    usage()
    path = sys.argv[1]

    # create directories for datasets if not exists
    for dir_path in ["data", "academia_data", "test_data"]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    tags_path = os.path.join(path, 'Tags.xml')
    tag_dict = Tag.parse_tags(tags_path) # 166,742 posts
    sorted_tags = sorted(tag_dict.items(), key=lambda x:x[1].count, reverse=True)

    for (tag_name, tag) in sorted_tags[:100]: print "{}: {}".format(tag_name, tag.count)

    # Generate a word cloud image
    wordcloud = WordCloud(background_color="white", width=1600, height=800).generate_from_frequencies([(tag_name, tag.count) for (tag_name, tag) in sorted_tags[:100]])

    # take relative word frequencies into account, lower max_font_size
    #wordcloud = WordCloud(max_font_size=40, relative_scaling=.5).generate(text)
    plt.figure()
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()

    # The pil way (if you don't have matplotlib)
    #image = wordcloud.to_image()
    #image.show()

if __name__ == "__main__":
    main()
