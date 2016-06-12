# -*- coding: utf-8 -*-

from xml.dom import minidom

class Tag(object):
    def __init__(self, name, count):
        self.name = name
        self.preprocessed_tag_name = None
        self.count = count


    @classmethod
    def parse_tags(cls, file_path):
        xmldoc = minidom.parse(file_path)
        item_list = xmldoc.getElementsByTagName('row')
        tag_dict = {}
        for s in item_list:
            tag_name = s.attributes['TagName'].value
            tag_count = int(s.attributes['Count'].value)
            tag_dict[tag_name] = cls(tag_name, tag_count)
        return tag_dict


    def __repr__(self):
        return "{}#{}".format(self.name, self.count)


    @staticmethod
    def update_tag_counts_according_to_posts(tags, posts):
        # reset count
        for tag in tags:
            tag.count = 0
        # recount
        for post in posts:
            for tag in post.tag_set:
                tag.count += 1

    @staticmethod
    def sort_tags_by_frequency(tags, reverse=True):
        return sorted(tags, key=lambda x: x.count, reverse=reverse)

