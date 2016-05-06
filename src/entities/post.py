from xml.dom import minidom

class Post(object):
    def __init__(self, pid, title, body, tags):
        self.pid = pid
        self.title = title
        self.body = body
        self.tags = tags
        self.body_tokens = None
        self._tag_names = [tag.name for tag in tags]

    @classmethod
    def parse_posts(cls, file_path, tag_dict):
        xmldoc = minidom.parse(file_path)
        itemlist = xmldoc.getElementsByTagName('row')
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
