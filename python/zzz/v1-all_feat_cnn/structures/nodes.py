

###########################################################
class TokenNode(object):
    def __init__(self, id, word_form, pos_tag=None, ner_tag=None, dep_label=None, dep_head=None, scene=None, utterance=None):
        self.id = int(id)
        self.word_form = str(word_form)

        self.pos_tag = str(pos_tag)
        self.ner_tag = str(ner_tag)

        self.dep_label = str(dep_label)
        self.dep_head = dep_head

        self._scene = scene
        self._utterance = utterance

    def __lt__(self, other):
        return self.id < other.id

    def __gt__(self, other):
        return other.__lt__(self)

    def parent_utterance(self):
        return self._utterance

    def parent_scene(self):
        return self._scene


###########################################################
class MentionNode(object):
    def __init__(self, id, tokens, referent, embedding=None, feature=None):
        self.id = id
        self.tokens = tokens
        self.referent = str(referent).lower()

        self.feature = feature
        self.embedding = embedding

    def __lt__(self, other):
        return self.id < other.id

    def __gt__(self, other):
        return other.__lt__(self)

    def __hash__(self):
        return hash(self.id)
