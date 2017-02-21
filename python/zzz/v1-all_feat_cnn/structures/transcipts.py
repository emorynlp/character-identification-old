

class Episode(object):
    def __init__(self, id, scenes=list(), previous=None, next=None):
        self.id = int(id)
        self.scenes = scenes

        self._previous = previous
        self._next = next

    def __lt__(self, other):
        return self.id < other.id

    def __gt__(self, other):
        return other.__lt__(self)

    def previous_episode(self):
        return self._previous

    def next_episode(self):
        return self._next


###########################################################
class Scene(object):
    def __init__(self, id, utterances=list(), episode=None, previous=None, next=None):
        self.id = int(id)
        self.utterances = utterances

        self._episode = episode

        self._previous = previous
        self._next = next

    def __lt__(self, other):
        return self.id < other.id

    def __gt__(self, other):
        return other.__lt__(self)

    def previous_scene(self):
        return self._previous

    def next_scene(self):
        return self._next

    def parent_episode(self):
        return self._episode


###########################################################
class Utterance(object):
    def __init__(self, speaker, utterances=list(), statements=list(), scene=None, previous=None, next=None):
        self.speaker = str(speaker).lower()
        self.utterances = utterances
        self.statements = statements

        self._scene = scene

        self._previous = previous
        self._next = next

    def previous_utterance(self):
        return self._previous

    def next_utterance(self):
        return self._next

    def parent_scene(self):
        return self._scene
