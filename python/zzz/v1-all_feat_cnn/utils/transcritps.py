

class TranscriptUtils(object):
    @staticmethod
    def collect_speakers(episodes, lowercase=True):
        speakers = set()
        for episode in episodes:
            for scene in episode.scenes:
                for utterance in scene.utterances:
                    speaker = utterance.speaker.lower() if lowercase else utterance.speaker
                    speakers.add(speaker)
        return speakers

    @staticmethod
    def collect_pos_tags(episodes):
        pos_tags = set()
        for episode in episodes:
            for scene in episode.scenes:
                for utterance in scene.utterances:
                    for ns in utterance.utterances:
                        for n in ns:
                            pos_tags.add(n.pos_tag)
                    for ns in utterance.statements:
                        for n in ns:
                            pos_tags.add(n.pos_tag)
        return pos_tags

    @staticmethod
    def collect_ner_tags(episodes):
        ner_tags = set()
        for episode in episodes:
            for scene in episode.scenes:
                for utterance in scene.utterances:
                    for ns in utterance.utterances:
                        for n in ns:
                            ner_tags.add(n.ner_tag)
                    for ns in utterance.statements:
                        for n in ns:
                            ner_tags.add(n.ner_tag)
        return ner_tags

    @staticmethod
    def collect_dep_labels(episodes):
        dep_labels = set()
        for episode in episodes:
            for scene in episode.scenes:
                for utterance in scene.utterances:
                    for ns in utterance.utterances:
                        for n in ns:
                            dep_labels.add(n.dep_label)
                    for ns in utterance.statements:
                        for n in ns:
                            dep_labels.add(n.dep_label)
        return dep_labels
