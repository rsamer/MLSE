# -*- coding: utf-8 -*-

from xml.dom import minidom

class Post(object):


    def __init__(self, pid, title, body, tag_set, score):
        assert isinstance(tag_set, set)
        self.pid = pid
        self.title = title
        self.body = body
        self.tag_set = tag_set
        self.tag_set_prediction = None
        self.score = score
        self.answers = None
        self.tokens = None
        self.tokens_pos_tags = None


    @classmethod
    def parse_posts(cls, file_path, tag_dict):
        xmldoc = minidom.parse(file_path)
        item_list = xmldoc.getElementsByTagName('row')
        posts = []
        for s in item_list:
            tag_set = set()
            if int(s.attributes["PostTypeId"].value) != 1:
                # TODO: link answers with posts
                continue

            for tag_name in s.attributes['Tags'].value.replace("<", "").split(">")[:-1]:
                if tag_name not in tag_dict:
                    raise RuntimeError("Tag '%s' not found!" % tag_name)
                tag = tag_dict[tag_name]
                if tag in tag_set:
                    raise RuntimeError("Tag duplicate!")
                tag_set.add(tag)

            pid = int(s.attributes['Id'].value)
            title = s.attributes['Title'].value
            body = s.attributes['Body'].value
            score = int(s.attributes['Score'].value)
            posts += [cls(pid, title, body, tag_set, score)]
        return posts


    def contains_tag_with_name(self, tag_name):
        for tag in self.tag_set:
            if tag.name == tag_name:
                return True
        return False


    def remove_tag_set(self, tag_set_to_be_removed):
        assert isinstance(tag_set_to_be_removed, set)
        self.tag_set -= tag_set_to_be_removed


    @staticmethod
    def copied_new_counted_tags_for_posts(posts):
        ''' For each tag: Counts how many times the tag occurs within the given post list
                          and updates the tag instances (used for clustering)
        '''
        import copy
        tag_dict = {}
        for post in posts:
            for tag in post.tag_set:
                if tag.name not in tag_dict:
                    new_tag = copy.copy(tag)
                    new_tag.count = 0
                    tag_dict[tag.name] = new_tag
                tag_dict[tag.name].count += 1
        return tag_dict.values()


    def __repr__(self):
        return "Post({},score={},'{}',tag_set='{}'".format(self.pid, self.score, self.title[:10], self.tag_set)
