from abc import *
import numpy as np
from utils.strings import lcs
from utils.evaluators import AbstractEvaluator


###########################################################
class AbstractFeatureExtractor(object):
    @abstractmethod
    def extract(self, object):
        return


###########################################################
class EntityFeatureExtractor(AbstractFeatureExtractor):
    def __init__(self, empty_embd_shape=None, empty_feat_shape=None):
        self.e_EMPTY = np.zeros(empty_embd_shape) if empty_embd_shape else None
        self.f_EMPTY = np.zeros(empty_feat_shape) if empty_feat_shape else None

    def extract(self, entity, include_average=True, nb_mentions=5, selection_method='last'):
        embedding, feature = ([], [])
        if entity and include_average:
            nb_mentions -= 1
            embedding.append(entity.get_avg_mention_embedding())
            feature.append(entity.get_avg_mention_feature())

        nb_padding = max(0, nb_mentions - len(entity))
        nb_mentions -= nb_padding

        if selection_method is 'last':
            mentions = entity[-nb_mentions:]
            embedding += map(lambda m: m.embedding, mentions)
            feature += map(lambda m: m.feature, mentions)

        for i in xrange(nb_padding):
            embedding.append(self.e_EMPTY)
            feature.append(self.f_EMPTY)

        return np.array(embedding), np.array(feature)


###########################################################
class MentionPairFeatureExtractor(AbstractFeatureExtractor):
    def extract(self, mention_pair):
        prev, curr = mention_pair
        p_tokens = " ".join(map(lambda t: t.word_form, prev.tokens))
        c_tokens = " ".join(map(lambda t: t.word_form, curr.tokens))

        match = float(len(lcs(p_tokens, c_tokens)))
        r1, r2 = match / len(p_tokens), match / len(c_tokens)

        return np.array([r1, r2])


###########################################################
def get_mention_metadata(mention):
    h_token, fst_token, lst_token = get_head_token(mention), mention.tokens[0], mention.tokens[-1]
    utterance = fst_token.parent_utterance()
    scene = utterance.parent_scene()
    episode = scene.parent_episode()

    for st_idx, statement in enumerate(utterance.statements):
        if h_token in statement and fst_token in statement and lst_token in statement:
            return episode, scene, utterance, st_idx, statement.index(h_token), h_token, \
                   statement.index(fst_token), fst_token, statement.index(lst_token), lst_token
    return None


def get_head_token(mention):
    tids = [t.id for t in mention.tokens]
    for token in mention.tokens:
        if token.dep_head is not None and token.dep_head.id not in tids:
            return token
    return mention.tokens[0]


def flatten_utterance_statements(utterance):
    return [st for statements in utterance.statements for st in statements]


class MentionFeatureExtractor(AbstractFeatureExtractor):
    def __init__(self, word2vec, word2gender, spks, poss, deps, ners, spk_dim=5, pos_dim=5, dep_dim=5, ner_dim=5):
        self.word2vec, self.word2vec_dim = word2vec, len(word2vec.values()[0])
        self.word2gender, self.word2gender_dim = word2gender, len(word2gender.values()[0])

        self.spk_dim, self.spk2vec = spk_dim, dict([(spk, np.random.rand(spk_dim)) for spk in spks])
        self.pos_dim, self.pos2vec = pos_dim, dict([(pos, np.random.rand(pos_dim)) for pos in poss])
        self.dep_dim, self.dep2vec = dep_dim, dict([(dep, np.random.rand(dep_dim)) for dep in deps])
        self.ner_dim, self.ner2vec = ner_dim, dict([(ner, np.random.rand(ner_dim)) for ner in ners])

    def extract(self, mention):
        _, _, curr_utterance, st_idx, th_idx, h_token, ts_idx, fst_token, te_idx, lst_token = get_mention_metadata(mention)

        tokens, speaker = mention.tokens, curr_utterance.speaker
        prev1_utterance = curr_utterance.previous_utterance()
        prev_speaker = prev1_utterance.speaker if prev1_utterance is not None else None
        next1_utterance = curr_utterance.next_utterance()
        next_speaker = next1_utterance.speaker if next1_utterance is not None else None

        prev2_utterance = prev1_utterance.previous_utterance() if prev1_utterance is not None else None
        next2_utterance = next1_utterance.previous_utterance() if next1_utterance is not None else None

        flatten_statements_tokens = flatten_utterance_statements(curr_utterance)
        sftid = flatten_statements_tokens.index(fst_token)
        eftid = flatten_statements_tokens.index(lst_token)

        emb1, emb2, emb3, emb4 = [], [], [], []
        # Group 1 embedding (Mention tokens, up to 4 tokens)
        for idx in xrange(4):
            emb1.append(self.get_token_word_vector(tokens[idx] if len(tokens) > idx else None))
        emb1 = np.array(emb1)

        # Group 2 embedding (+-# token sequence)
        emb2.append(self.get_tokens_word_vector(flatten_statements_tokens, sftid - 3, 1))  # -3 word from the mention
        emb2.append(self.get_tokens_word_vector(flatten_statements_tokens, sftid - 2, 1))  # -2 word from the mention
        emb2.append(self.get_tokens_word_vector(flatten_statements_tokens, sftid - 1, 1))  # -1 word from the mention
        emb2.append(self.get_mention_words_vector(mention))                                # Avg of words in the mention
        emb2.append(self.get_tokens_word_vector(flatten_statements_tokens, eftid + 1, 1))  # +1 word from the mention
        emb2.append(self.get_tokens_word_vector(flatten_statements_tokens, eftid + 2, 1))  # +2 word from the mention
        emb2.append(self.get_tokens_word_vector(flatten_statements_tokens, eftid + 3, 1))  # +2 word from the mention
        emb2 = np.array(emb2)

        # Group 3 embedding (Sentence vector)
        prev2_sentence = curr_utterance.statements[st_idx - 2] if st_idx-1 > 0 else None
        prev1_sentence = curr_utterance.statements[st_idx - 1] if st_idx   > 0 else None
        next1_sentence = curr_utterance.statements[st_idx + 1] if st_idx   < len(curr_utterance.statements)-1 else None
        next2_sentence = curr_utterance.statements[st_idx + 2] if st_idx+1 < len(curr_utterance.statements)-1 else None
        emb3.append(self.get_tokens_word_vector(prev2_sentence, 0, len(prev2_sentence))
                    if prev2_sentence is not None else np.zeros(self.word2vec_dim))  # -2 sentence from the mention
        emb3.append(self.get_tokens_word_vector(prev1_sentence, 0, len(prev1_sentence))
                    if prev1_sentence is not None else np.zeros(self.word2vec_dim))  # -1 sentence from the mention
        emb3.append(self.get_tokens_word_vector(curr_utterance.statements[st_idx], 0, len(curr_utterance.statements[st_idx])))
        emb3.append(self.get_tokens_word_vector(next1_sentence, 0, len(next1_sentence))
                    if next1_sentence is not None else np.zeros(self.word2vec_dim))  # +1 sentence from the mention
        # emb3.append(self.get_tokens_word_vector(next2_sentence, 0, len(next2_sentence))
        #             if next2_sentence is not None else np.zeros(self.word2vec_dim))  # +2 sentence from the mention
        emb3 = np.array(emb3)

        # Group 4 embedding (Utterance vector)
        emb4.append(self.get_utterance_vector(prev2_utterance))  # -2 utterance from the mention
        emb4.append(self.get_utterance_vector(prev1_utterance))  # -1 utterance from the mention
        emb4.append(self.get_utterance_vector(curr_utterance))   # current utterance from the mention
        emb4.append(self.get_utterance_vector(next1_utterance))  # +1 utterance from the mention
        # emb4.append(self.get_utterance_vector(next2_utterance))  # +2 utterance from the mention
        emb4 = np.array(emb4)

        features = list()
        features.append(self.get_tokens_gender_vector(mention))  # Avg gender information of the mention
        features.append(self.get_speaker_vector(prev_speaker))   # Previous speaker information of the utterance
        features.append(self.get_speaker_vector(speaker))        # Current speaker information of the utterance
        # features.append(self.get_speaker_vector(next_speaker))   # Next speaker information of the utterance

        # # Gender information of head token in the mention
        # features.append(self.get_token_gender_vector(h_token))
        # # Pos tag information of head token
        # features.append(self.get_pos_tag_vector(h_token.pos_tag))
        # # Ner tag information of head token
        # features.append(self.get_ner_tag_vector(h_token.ner_tag))
        # # Dep label information of head token
        # features.append(self.get_dep_label_vector(h_token.dep_label))
        # # Dep label information of head token'parent
        # features.append(np.zeros(self.dep_dim) if h_token.dep_head is None
        #                 else self.get_dep_label_vector(h_token.dep_head.dep_label))
        features = np.concatenate(features)

        return [emb1, emb2, emb3, emb4], features

    ###### Mention tokens features #######
    def get_token_word_vector(self, token):
        if token is not None:
            word_form = token.word_form
            return self.word2vec[word_form] if word_form in self.word2vec else np.zeros(self.word2vec_dim)
        return np.zeros(self.word2vec_dim)

    def get_mention_words_vector(self, mention):
        tvector = np.zeros(self.word2vec_dim)
        for token in mention.tokens:
            tvector += self.get_token_word_vector(token)
        return tvector / float(len(mention.tokens))

    def get_tokens_word_vector(self, flatten_tokens, start, length):
        tvector = np.zeros(self.word2vec_dim)
        for tid in xrange(start, start+length):
            tvector += self.get_token_word_vector(flatten_tokens[tid]) \
                if tid >=0 and tid < len(flatten_tokens) else np.zeros(self.word2vec_dim)
        return tvector / float(length)

    def get_token_gender_vector(self, token):
        word_form = token.word_form.lower()
        return self.word2gender[word_form] if word_form in self.word2gender else np.zeros(self.word2gender_dim)

    def get_tokens_gender_vector(self, mention):
        gvector = np.zeros(self.word2gender_dim)
        for token in mention.tokens:
            gvector += self.get_token_gender_vector(token)
        return gvector / float(len(mention.tokens))

    def get_speaker_vector(self, speaker):
        return self.spk2vec[speaker] if speaker in self.spk2vec else np.zeros(self.spk_dim)

    def get_pos_tag_vector(self, tag):
        return self.pos2vec[tag] if tag in self.pos2vec else np.zeros(self.pos_dim)

    def get_ner_tag_vector(self, tag):
        return self.ner2vec[tag] if tag in self.ner2vec else np.zeros(self.ner_dim)

    def get_dep_label_vector(self, label):
        return self.dep2vec[label] if label in self.dep2vec else np.zeros(self.dep_dim)

    #### Transcript document features ####
    def get_utterance_vector(self, utterance):
        uvector = np.zeros(self.word2vec_dim)
        if utterance is not None:
            for st in utterance.statements:
                uvector += self.get_tokens_word_vector(st, 0, len(st))
            return uvector / float(len(utterance.statements)) if utterance.statements else uvector
        return uvector

    def get_scene_vector(self, scene):
        svector = np.zeros(self.word2vec_dim)
        if scene is not None:
            for utterance in scene.utterances:
                svector += self.get_utterance_vector(utterance)
            return svector / float(len(scene.utterances)) if scene.utterances else svector
        return svector

    def get_episode_vector(self, episode):
        evector = np.zeros(self.word2vec_dim)
        if episode is not None:
            for scene in episode.scenes:
                evector += self.get_scene_vector(scene)
            return evector / float(len(episode.scenes)) if episode.scenes else evector
        return evector
