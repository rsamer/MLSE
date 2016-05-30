# -*- coding: utf-8 -*-

import logging
from xml.dom import minidom

_logger = logging.getLogger(__name__)


class Answer(object):
    def __init__(self, pid, body, score):
        self.pid = pid
        self.body = body
        self.score = score

    def __repr__(self):
        return "Answer({},score={},'{}'".format(self.pid, self.score, self.body[:20].encode("utf8"))


class Post(object):

    def __init__(self, pid, title, body, tag_set, score, accepted_answer_id=None, answers=[]):
        assert isinstance(tag_set, set)
        assert isinstance(answers, list)
        self.pid = pid
        self.title = title
        self.body = body
        self.tag_set = tag_set
        self.tag_set_prediction = None
        self.score = score
        self.accepted_answer_id = accepted_answer_id
        self.answers = answers
        self.tokens = None
        self.tokens_pos_tags = None

    @classmethod
    def parse_posts(cls, file_path, tag_dict):
        xmldoc = minidom.parse(file_path)
        item_list = xmldoc.getElementsByTagName('row')
        posts = {}
        grouped_answers = {}
        for s in item_list:
            tag_set = set()
            pid = int(s.attributes['Id'].value)
            score = int(s.attributes['Score'].value)
            post_type_id = int(s.attributes["PostTypeId"].value)
            body = s.attributes['Body'].value
            '''
            Explanation Post Type IDs:
            --------------------------
            1 Question
            2 Answer
            3 Wiki
            4 TagWikiExcerpt
            5 TagWiki
            6 ModeratorNomination
            7 WikiPlaceholder
            8 PrivilegeWiki
            '''
            if post_type_id == 2:
                question_id = int(s.attributes["ParentId"].value)
                if question_id not in grouped_answers:
                    grouped_answers[question_id] = []
                grouped_answers[question_id] += [Answer(pid, body, score)]
                continue
            elif post_type_id >= 3:
                continue # omit

            assert post_type_id == 1
            for tag_name in s.attributes['Tags'].value.replace("<", "").split(">")[:-1]:
                if tag_name not in tag_dict:
                    raise RuntimeError("Tag '%s' not found!" % tag_name)
                tag = tag_dict[tag_name]
                if tag in tag_set:
                    _logger.warn("Multiple tag assignments for tag '{}' in post with ID {}".format(tag_name, pid))
                tag_set.add(tag)

            title = s.attributes['Title'].value
            accepted_answer_id = int(s.attributes["AcceptedAnswerId"].value) if s.hasAttribute("AcceptedAnswerId") else None
            posts[pid] = cls(pid, title, body, tag_set, score, accepted_answer_id, [])

        # finally add answers to corresponding posts
        for question_id, answers in grouped_answers.iteritems():
            if question_id not in posts:
                _logger.info("Ignoring answer for non existant post {}".format(question_id))
                continue
            corresponding_post = posts[question_id]
            corresponding_post.answers = answers

        return posts.values()

    def contains_tag_with_name(self, tag_name):
        for tag in self.tag_set:
            if tag.name == tag_name:# in tag.name:
                return True
        return False

    def remove_tag_set(self, tag_set_to_be_removed):
        assert isinstance(tag_set_to_be_removed, set)
        self.tag_set -= tag_set_to_be_removed

    def accepted_answer(self):
        if not self.accepted_answer_id: return None
        accepted_answers = filter(lambda a: self.accepted_answer_id == a.pid, self.answers)
        assert len(accepted_answers) <= 1, "There must be one accepted answer!"
        if len(accepted_answers) == 0:
            return None # TODO: rempve from example dataset...
        return accepted_answers[0]

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
        return "Post({},score={},#answers={},'{}',tag_set='{}'".format(self.pid, self.score, len(self.answers), self.title[:10].encode("utf8"), self.tag_set)
