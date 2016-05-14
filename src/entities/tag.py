# -*- coding: utf-8 -*-

from xml.dom import minidom

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

    @staticmethod
    def sort_tags_by_frequency(tags, reverse=True):
        return sorted(tags, key=lambda x: x.count, reverse=True)
