import numpy as np
import re, json, struct
from structures.nodes import *
from structures.transcipts import *


###########################################################
class TranscriptJsonReader(object):
    @staticmethod
    def read_season(file):
        return TranscriptJsonReader()._read_season(file)

    def _read_season(self, file):
        season = list()
        utterance_mentions = list()
        statement_mentions = list()
        raw_json = json.load(file)

        episodes_json = raw_json["episodes"]
        episode_ids = list(map(lambda x: int(x), episodes_json.keys()))

        prev_episode = None
        for eid in range(min(episode_ids), max(episode_ids)+1):

            if eid in episode_ids:
                curr_episode = self._parse_episode_json(episodes_json[str(eid)], utterance_mentions, statement_mentions)
                self._assign_metadata(curr_episode)

                season.append(curr_episode)
                if prev_episode is not None:
                    prev_episode._next = curr_episode
                    curr_episode._previous = prev_episode
                prev_episode = curr_episode
            else:
                prev_episode = None

        return season, utterance_mentions, statement_mentions

    def _parse_episode_json(self, episode_json, utterance_mentions, statement_mentions):
        scenes = list()
        eid = int(episode_json["episode_id"])

        scenes_json = episode_json["scenes"]
        scene_ids = list(map(lambda x: int(x), scenes_json.keys()))

        prev_scene = None
        for sid in range(min(scene_ids), max(scene_ids) + 1):

            if sid in scene_ids:
                curr_scene = self._parse_scene_json(scenes_json[str(sid)], utterance_mentions, statement_mentions)

                scenes.append(curr_scene)
                if prev_scene is not None:
                    prev_scene._next = curr_scene
                    curr_scene._previous = prev_scene
                prev_scene = curr_scene

            else:
                prev_scene = None

        return Episode(eid, scenes)

    def _parse_scene_json(self, scene_json, utterance_mentions, statement_mentions):
        utterances = list()
        sid = int(scene_json["scene_id"])

        utterances_json = scene_json["utterances"]

        prev_utterance = None
        for u_json in utterances_json:
            curr_utterance = self._parse_utterance_json(u_json, utterance_mentions, statement_mentions)
            utterances.append(curr_utterance)

            if prev_utterance is not None:
                prev_utterance._next = curr_utterance
                curr_utterance._previous = prev_utterance
            prev_utterance = curr_utterance

        return Scene(sid, utterances)

    def _parse_utterance_json(self, u_json, utterance_mentions, statement_mentions):
        speaker = u_json["speaker"]
        utterances = list()
        statements = list()

        utterance_word_forms = u_json["tokenized_utterance_sentences"]
        statement_word_forms = u_json["tokenized_statement_sentences"]

        utterance_pos_tags = u_json.get("utterance_sentence_pos_tags", None)
        statement_pos_tags = u_json.get("statement_sentence_pos_tags", None)

        utterance_ner_tags = u_json.get("utterance_sentence_ner_tags", None)
        statement_ner_tags = u_json.get("statement_sentence_ner_tags", None)

        utterance_dep_labels = u_json.get("utterance_sentence_dep_labels", None)
        statement_dep_labels = u_json.get("statement_sentence_dep_labels", None)

        utterance_dep_heads = u_json.get("utterance_sentence_dep_heads", None)
        statement_dep_heads = u_json.get("statement_sentence_dep_heads", None)

        utterance_annotations = u_json.get("utterance_sentence_annotations", None)
        statement_annotations = u_json.get("statement_sentence_annotations", None)

        for idx, word_forms in enumerate(utterance_word_forms):
            utterance_nodes = self._parse_token_nodes(word_forms, utterance_pos_tags[idx], utterance_ner_tags[idx],
                                                      utterance_dep_labels[idx], utterance_dep_heads[idx])
            utterances.append(utterance_nodes)
            self._parse_mention_nodes(utterance_mentions, utterance_nodes, utterance_annotations[idx])

        for idx, word_forms in enumerate(statement_word_forms):
            statement_nodes = self._parse_token_nodes(word_forms, statement_pos_tags[idx], statement_ner_tags[idx],
                                                      statement_dep_labels[idx], statement_dep_heads[idx])
            statements.append(statement_nodes)
            self._parse_mention_nodes(statement_mentions, statement_nodes, statement_annotations[idx])

        return Utterance(speaker, utterances, statements)

    def _parse_token_nodes(self, word_forms, pos_tags, ner_tags, dep_labels, dep_heads):
        nodes = list()

        for idx, word_form in enumerate(word_forms):
            pos_tag = pos_tags[idx] if pos_tags is not None else None
            ner_tag = ner_tags[idx] if ner_tags is not None else None
            dep_label = dep_labels[idx] if dep_labels is not None else None

            nodes.append(TokenNode(int(idx+1), str(word_form), str(pos_tag), str(ner_tag), str(dep_label)))

        if dep_heads is not None:
            dep_heads = map(lambda x: int(x), dep_heads)
            for idx, head_id in enumerate(dep_heads):
                if head_id > 0:
                    nodes[idx].dep_head = nodes[head_id]

        return nodes

    def _parse_mention_nodes(self, mentions, token_nodes, referent_annotations):
        mention = None
        for idx, annotation in enumerate(referent_annotations):
            bilou = str(annotation[0])

            if bilou in ['B', 'U']:
                referent = str(annotation[2:])
                mention = MentionNode(len(mentions), [token_nodes[idx]], referent)

                mentions.append(mention)

            elif bilou in ['I', 'L'] and mention is not None:
                mention.tokens.append(token_nodes[idx])

    def _assign_metadata(self, episode):
        for scene in episode.scenes:
            scene._episode = episode

            for utterance in scene.utterances:
                utterance._scene = scene

                for u in utterance.utterances:
                    for n in u:
                        n._scene = scene
                        n._utterance = utterance

                for s in utterance.statements:
                    for n in s:
                        n._scene = scene
                        n._utterance = utterance


###########################################################
class Word2VecReader(object):
    @staticmethod
    def load_bin(file):
        return Word2VecReader()._load_bin(file)

    def _load_bin(self, file):
        word2vec = dict()

        vsize, dim = map(lambda x: int(x), file.readline().split())
        for idx in range(vsize):
            vocab = self._read_word(file)
            vector = np.array(struct.unpack('<'+'f'*dim, file.read(4*dim)))
            struct.unpack('c', file.read(1))    # Read off '\n'

            word2vec[vocab] = (vector / np.linalg.norm(vector)).astype('float32')

        return word2vec

    def _read_word(self, stream):
        buf = []
        while True:
            c = struct.unpack('c', stream.read(1))[0]
            if c not in [' ', '\n']:
                buf.append(c)
            else:
                break

        return "".join(buf)


###########################################################
# Male/Female/Neutral
class GenderDataReader(object):
    @staticmethod
    def load(file, word_only=False, normalize=False):
        return GenderDataReader()._load(file, word_only, normalize)

    def _load(self, file, word_only=False, normalize=False):
        word2gender = dict()

        word_regex = re.compile('^[A-Za-z]+$')
        for line in file.readlines():
            string, data = line.lower().split('\t')
            string = string.replace("!", "").strip()

            if word_only and word_regex.match(string) is None:
                continue

            vector = map(lambda x: int(x), data.split()[:3])
            tcount = float(sum(vector))

            vector = np.array(vector) if not normalize or tcount == 0.0 else np.array(vector) / tcount
            word2gender[string] = vector.astype('float32')

        return word2gender
