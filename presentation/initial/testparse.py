#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os
from xml.dom import minidom
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pandas as pd
import seaborn as sns
import re
from bs4 import BeautifulSoup

class Tag(object):
    def __init__(self, name, count):
        self.name = name
        self.count = count

    @classmethod
    def parse_tags(cls, file_path):
        xmldoc = minidom.parse(file_path)
        itemlist = xmldoc.getElementsByTagName('row')
        tag_dict = {}
        for s in itemlist:
            tag_name = s.attributes['TagName'].value
            tag_count = int(s.attributes['Count'].value)
            tag_dict[tag_name] = cls(tag_name, tag_count)
        return tag_dict

    def __repr__(self):
        return "{}#{}".format(self.name, self.count)

class Post(object):
    def __init__(self, pid, title, body, tags):
        self.pid = pid
        self.title = title
        self.body = body
        self.tags = tags
        self._tag_names = [tag.name for tag in tags]

    @classmethod
    def parse_posts(cls, file_path, tag_dict):
        xmldoc = minidom.parse(file_path)
        itemlist = xmldoc.getElementsByTagName('row')
        num_of_posts = len(itemlist)
        posts = []
        for s in itemlist:
            tags = set()
            if int(s.attributes["PostTypeId"].value) != 1:
                # TODO: link answers with posts
                continue

            for tag_name in s.attributes['Tags'].value.replace("<", "").split(">")[:-1]:
                if tag_name not in tag_dict:
                    raise RuntimeError("Tag '%s' not found!" % tag_name)
                tag = tag_dict[tag_name]
                if tag in tags:
                    raise RuntimeError("Tag duplicate!")
                tags.add(tag)

            pid = s.attributes['Id'].value
            title = s.attributes['Title'].value
            body = s.attributes['Body'].value
            posts += [cls(pid, title, body, tags)]
        return posts

    def tag_exists(self, tag_name):
        return tag_name in self._tag_names

    def __repr__(self):
        return "Post({},'{}',tags='{}'".format(self.pid, self.title[:10], self.tags)


def plot_bar_chart(data_labels, data_points, ylabel, title, bar_width, filename=None):
    ind = np.arange(len(data_points))  # the x locations for the groups

    rcParams.update({'figure.autolayout': True})
    fig, ax = plt.subplots()

    rects1 = ax.bar(ind, data_points, bar_width, color='b')
    plt.xticks(rotation=70)

    # add some text for labels, title and axes ticks
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(ind + bar_width/2.)
    ax.set_xticklabels(data_labels)

    def autolabel(rects):
        # attach some text labels
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                    '%.3f' % round(height, 3),
                    ha='center', va='bottom')
    autolabel(rects1)

    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()

def plot_line_chart(x_points, y_points, x_label, y_label, title, filename=None, highlight_points=[]):
    plt.clf()
    plt.plot(x_points, y_points)
    plt.xlabel(x_label)
    plt.yticks([0., 0.25, 0.5, 0.75, 0.9, 1.])
    plt.ylabel(y_label)
    plt.title(title)

    for highlight_point in highlight_points:
        plt.plot(highlight_point[0], highlight_point[1], color='r', linewidth=1)
    
    highlight_ticks = [point for sublist_points in highlight_points for point in sublist_points[0]]
    custom_ticks = range(100, 1700, 200)
    plt.xticks(custom_ticks + highlight_ticks)

    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()

def plot_correlation_chart(matrix, tag_names, size, filename=None):
    sns.set(style="white")
    assert len(matrix) > 0
    num_columns = len(matrix[0])
    assert size <= num_columns

    # Generate a large random dataset
    d = pd.DataFrame(data=matrix,
                     columns=tag_names)
    # Compute the correlation matrix
    corr = d.corr()
    corr.drop(corr.columns[range(size, num_columns)], axis=1, inplace=True)
    corr.drop(corr.index[range(size, num_columns)], inplace=True)

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    #plt.gca().tight_layout()
    plt.gcf().subplots_adjust(bottom=0.2)
    #rcParams.update({'figure.autolayout': True})
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        vmax=.3,
        square=True,
        linewidths=.5,
        cbar_kws={"shrink": .5},
        ax=ax
    )
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)

    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()

def main():
    if len(sys.argv) != 2:
        raise RuntimeError("Invalid arguments given")
    path = sys.argv[1]

    tags_path = os.path.join(path, 'Tags.xml')
    tag_dict = Tag.parse_tags(tags_path) # 166,742 posts
    sorted_tags = sorted(tag_dict.items(), key=lambda x:x[1].count, reverse=True)
    expected_total_tag_assignment_size = reduce(lambda x,y: x+y, [tag.count for (_, tag) in sorted_tags])
    all_tag_names = [tag_name for (tag_name, _) in sorted_tags]
    top_10_tag_names = [tag_name for (tag_name, _) in sorted_tags[:10]]
    top_20_tag_names = [tag_name for (tag_name, _) in sorted_tags[:20]]
    top_30_tag_names = [tag_name for (tag_name, _) in sorted_tags[:30]]
    top_10_tag_frequencies = [tag.count/float(expected_total_tag_assignment_size) for (_, tag) in sorted_tags[:10]]
    #plot_bar_chart(top_10_tag_names, top_10_tag_frequencies, "Tag frequency", "", 0.7, "tag-frequency-top-10.png")
    plot_bar_chart(top_10_tag_names, top_10_tag_frequencies, "Tag frequency", "", 0.7)

    y_points = [tag.count/float(expected_total_tag_assignment_size) for (_, tag) in sorted_tags[:300]]
    x_points = range(1, len(y_points) + 1)
    #plot_line_chart(x_points, y_points, "Top k tags", "Tag frequency", "Top 300 tags", filename="top-300-tags.png")
    plot_line_chart(x_points, y_points, "Top k tags", "Tag frequency", "Top 300 tags")

    current_tag_count = 0
    current_tag = 0
    y_points = []
    x_points = []
    highlight_points = []
    for (_, tag) in sorted_tags:
        current_tag_count += tag.count
        current_tag_count_normalized = current_tag_count/float(expected_total_tag_assignment_size)
        y_points += [current_tag_count_normalized]
        current_tag += 1
        x_points += [current_tag]

        if abs(current_tag_count_normalized - 0.5) <= 0.001 and len(highlight_points) == 0:
            highlight_points += [[[current_tag, current_tag], [0, current_tag_count_normalized]]]
            highlight_points += [[[0, current_tag], [current_tag_count_normalized, current_tag_count_normalized]]]

        if abs(current_tag_count_normalized - 0.75) <= 0.001 and len(highlight_points) == 2:
            highlight_points += [[[current_tag, current_tag], [0, current_tag_count_normalized]]]
            highlight_points += [[[0, current_tag], [current_tag_count_normalized, current_tag_count_normalized]]]

        if abs(current_tag_count_normalized - 0.9) <= 0.001 and len(highlight_points) == 4:
            highlight_points += [[[current_tag, current_tag], [0, current_tag_count_normalized]]]
            highlight_points += [[[0, current_tag], [current_tag_count_normalized, current_tag_count_normalized]]]

    #plot_line_chart(x_points, y_points, "Top k tags", "Cumulative tag frequency", "", "cumulative-tag-frequency.png", highlight_points)
    plot_line_chart(x_points, y_points, "Top k tags", "Cumulative tag frequency", "", highlight_points=highlight_points)

    posts_path = os.path.join(path, 'Posts.xml')
    posts = Post.parse_posts(posts_path, tag_dict)
    min_tag_size = None
    max_tag_size = 0
    total_tag_assignment_size = 0
    min_post_length = None
    max_post_length = 0
    total_post_length = 0
    min_post_word_count = None
    max_post_word_count = 0
    total_post_word_count = 0
    '''
    --------------------------
    |Post#|Tag1|Tag2|...|TagM|
    --------------------------
    |Post1|  0 |  1 |...|  0 |
    |Post2|  1 |  0 |...|  1 |
    |Post3|  1 |  1 |...|  1 |
    | ... |... |... |...|... |
    |PostN|  1 |  1 |...|  1 |
    --------------------------

    num_cols = len(top_20_tag_names) # M
    num_rows = len(posts) # N
    matrix = np.array([
        [0, 1, ..., 0],
        [1, 0, ..., 1],
        [1, 1, ..., 1],
        ...,
        [1, 1, ..., 1],
    ])
    '''
    matrix_list = []
    for post in posts:
        tag_size = len(post.tags)
        if min_tag_size is None or tag_size < min_tag_size:
            min_tag_size = tag_size
        if tag_size > max_tag_size:
            max_tag_size = tag_size
        total_tag_assignment_size += tag_size

        raw_post_body = BeautifulSoup(post.body, "html.parser").text
        post_length = len(raw_post_body)
        if min_post_length is None or post_length < min_post_length:
            min_post_length = post_length
        if post_length > max_post_length:
            max_post_length = post_length
        total_post_length += post_length

        post_word_count = len(re.findall(r'\b\w+\b', raw_post_body))
        if min_post_word_count is None or post_word_count < min_post_word_count:
            min_post_word_count = post_word_count
        if post_word_count > max_post_word_count:
            max_post_word_count = post_word_count
        total_post_word_count += post_word_count

        matrix_list.append([int(post.tag_exists(tag_name)) for tag_name in top_30_tag_names])
        #print post

    matrix = np.array(matrix_list)
    #plot_correlation_chart(matrix, top_30_tag_names, 30, "correlation-chart.png")
    plot_correlation_chart(matrix, top_30_tag_names, 30)

    assert sys.argv[1] != "data" or total_tag_assignment_size == expected_total_tag_assignment_size
    print "Number of posts: {}".format(len(posts))
    print "Number of tags: {}".format(len(tag_dict))
    for (tag_name, tag) in sorted_tags[:20]:
        print "{}: {}".format(tag_name, tag.count)
    print "Min tag/post: {}".format(min_tag_size)
    print "Max tag/post: {}".format(max_tag_size)
    print "Average tag size: {}".format(float(total_tag_assignment_size)/len(posts))
    print "Total tag assignment size: %d" % total_tag_assignment_size

    print "Min post char length: {}".format(min_post_length)
    print "Max post char length: {}".format(max_post_length)
    print "Average post char length: {}".format(float(total_post_length) / len(posts))
    print "Total post char length: %d" % total_post_length

    print "Min post word count: {}".format(min_post_word_count)
    print "Max post word count: {}".format(max_post_word_count)
    print "Average post word count: {}".format(float(total_post_word_count) / len(posts))
    print "Total post word count: %d" % total_post_word_count

if __name__ == "__main__":
    main()
