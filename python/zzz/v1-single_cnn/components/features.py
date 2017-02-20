from abc import *
import numpy as np


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
class MentionFeatureExtractor(AbstractFeatureExtractor):
    def __init__(self, word2vec, word2gender, spks, poss, deps, ners, spk_dim=8, pos_dim=8, dep_dim=8, ner_dim=8):
        self.word2vec = word2vec
        self.word2vec_dim = len(word2vec.values()[0])

        self.word2gender = word2gender
        self.word2gender_dim = len(word2gender.values()[0])

        self.spk_dim = spk_dim
        self.spk2vec = dict()
        for spk in spks:
            self.spk2vec[spk] = np.random.rand(spk_dim)

        self.pos_dim = pos_dim
        self.pos2vec = dict()
        for pos in poss:
            self.pos2vec[pos] = np.random.rand(pos_dim)

        self.dep_dim = dep_dim
        self.dep2vec = dict()
        for dep in deps:
            self.dep2vec[dep] = np.random.rand(dep_dim)

        self.ner_dim = ner_dim
        self.ner2vec = dict()
        for ner in ners:
            self.ner2vec[ner] = np.random.rand(ner_dim)

    def extract(self, mention):
        head_token = self.get_head_token(mention)
        first_token, last_token = mention.tokens[0], mention.tokens[-1]

        utterance = first_token.parent_utterance()
        scene = utterance.parent_scene()
        episode = scene.parent_episode()
        speaker = utterance.speaker

        prev_utterance = utterance.previous_utterance()
        prev_speaker = prev_utterance.speaker if prev_utterance is not None else None

        flatten_utterance_tokens = self.flatten_utterance(utterance)
        flatten_sentence_tokens = self.get_mention_sentence_tokens(utterance, mention)
        ft_locations = self.get_token_locations(flatten_utterance_tokens, mention)
        start_ftid, end_ftid = ft_locations[0], ft_locations[-1]
        token_len = end_ftid - start_ftid

        embeddings = list()
        # Word embeddings of the head word
        embeddings.append(self.get_token_word_vector(head_token))
        # First word of the mention
        embeddings.append(self.get_token_word_vector(first_token))
        # Last word of the mention
        embeddings.append(self.get_token_word_vector(last_token))
        # Avg of all words in the mention
        embeddings.append(self.get_tokens_word_vector(mention))
        # Two preceding words of the mention
        embeddings.append(self.get_tokens_word_vector_wOffset(flatten_utterance_tokens, start_ftid-1, 1))
        embeddings.append(self.get_tokens_word_vector_wOffset(flatten_utterance_tokens, start_ftid-2, 1))
        # Two following words of the mention
        embeddings.append(self.get_tokens_word_vector_wOffset(flatten_utterance_tokens, end_ftid+1, 1))
        embeddings.append(self.get_tokens_word_vector_wOffset(flatten_utterance_tokens, end_ftid+2, 1))
        # Avg of the +-1 words
        embeddings.append(self.get_tokens_word_vector_wOffset(flatten_utterance_tokens, start_ftid-1, token_len+2))
        # Avg of the +-2 words
        embeddings.append(self.get_tokens_word_vector_wOffset(flatten_utterance_tokens, start_ftid-2, token_len+4))
        # Avg of the -5 words
        embeddings.append(self.get_tokens_word_vector_wOffset(flatten_utterance_tokens, start_ftid-1, -5))
        # Avg of the +5 words
        embeddings.append(self.get_tokens_word_vector_wOffset(flatten_utterance_tokens, end_ftid+1, 5))
        # Avg of all words in the mention's sentence
        embeddings.append(self.get_tokens_word_vector_wOffset(flatten_sentence_tokens, 0, len(flatten_sentence_tokens)))
        # Avg of all words in current utterance
        embeddings.append(self.get_utterance_vector(utterance))
        # Avg of all words in previous utterance
        embeddings.append(self.get_utterance_vector(prev_utterance))
        # Avg of all words in the scene
        embeddings.append(self.get_scene_vector(scene))
        # Avg of all words in the episode
        embeddings.append(self.get_episode_vector(episode))

        features = list()
        # Gender information of head token in the mention
        features.append(self.get_token_gender_vector(head_token))
        # Avg gender information of all tokens in the mention
        features.append(self.get_tokens_gender_vector(mention))
        # Current speaker information of the utterance
        features.append(self.get_speaker_vector(speaker))
        # Previous speaker information of the utterance
        features.append(self.get_speaker_vector(prev_speaker))
        # Pos tag information of head token
        features.append(self.get_pos_tag_vector(head_token.pos_tag))
        # Ner tag information of head token
        features.append(self.get_ner_tag_vector(head_token.ner_tag))
        # Dep label information of head token
        features.append(self.get_dep_label_vector(head_token.dep_label))
        # Dep label information of head token'parent
        features.append(np.zeros(self.dep_dim) if head_token.dep_head is None
                        else self.get_dep_label_vector(head_token.dep_head.dep_label))
        # Mention token length/location information within utterance
        features.append(self.get_mention_location_information(flatten_utterance_tokens, start_ftid, end_ftid))

        return np.array(embeddings), np.concatenate(features)

    ###### Helper functions #######
    def get_head_token(self, mention):
        tids = map(lambda t: t.id, mention.tokens)
        for token in mention.tokens:
            if token.dep_head is not None and token.dep_head.id not in tids:
                return token
        return mention.tokens[0]

    def flatten_utterance(self, utterance):
        return [st for statements in utterance.statements for st in statements]

    def get_token_locations(self, flatten_tokens, mention):
        locations = []
        for idx, token in enumerate(flatten_tokens):
            if token in mention.tokens:
                locations.append(idx)
        locations.sort()
        return locations

    def get_mention_sentence_tokens(self, utterance, mention):
        token = mention.tokens[0]
        for statement in utterance.statements:
            if token in statement:
                return statement
        return None

    ###### Mention tokens features #######
    def get_token_word_vector(self, token):
        word_form = token.word_form.lower()
        return self.word2vec[word_form] if word_form in self.word2vec else np.zeros(self.word2vec_dim)

    def get_tokens_word_vector(self, mention):
        tvector = np.zeros(self.word2vec_dim)
        for token in mention.tokens:
            tvector += self.get_token_word_vector(token)
        return tvector / float(len(mention.tokens))

    def get_tokens_word_vector_wOffset(self, flatten_tokens, start, offset):
        tvector = np.zeros(self.word2vec_dim)

        if offset > 0:
            for tid in xrange(start, start+offset):
                tvector += self.get_token_word_vector(flatten_tokens[tid]) \
                    if tid < len(flatten_tokens) else np.zeros(self.word2vec_dim)
        else:
            for tid in xrange(start, start-offset, -1):
                tvector += self.get_token_word_vector(flatten_tokens[tid]) \
                    if tid <= 0 else np.zeros(self.word2vec_dim)
        return tvector / float(offset)

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

    def get_mention_location_information(self, flatten_utternace_tokens, start_idx, end_index):
        length = len(flatten_utternace_tokens)
        # Normalized mention word length, start token location, end token location
        return np.array([float(end_index-start_idx)/length, float(start_idx)/length, float(end_index)/length])

    #### Transcript document features ####
    def get_utterance_vector(self, utterance):
        tcount = 0
        uvector = np.zeros(self.word2vec_dim)
        if utterance is not None:
            for u in utterance.statements:
                for t in u:
                    word = t.word_form.lower()
                    if word in self.word2vec:
                        uvector = uvector + self.word2vec[word]
                tcount += len(u)
        return uvector / float(tcount) if tcount > 0 else uvector

    def get_scene_vector(self, scene):
        svector = np.zeros(self.word2vec_dim)
        for utterance in scene.utterances:
            svector += self.get_utterance_vector(utterance)

        return svector / float(len(scene.utterances)) if scene.utterances else svector

    def get_episode_vector(self, episode):
        evector = np.zeros(self.word2vec_dim)
        for scene in episode.scenes:
            evector += self.get_scene_vector(scene)

        return evector / float(len(episode.scenes)) if episode.scenes else evector
