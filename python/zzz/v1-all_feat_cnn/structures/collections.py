import numpy as np


class MentionCluster(list):
    def __init__(self, *args):
        list.__init__(self, *args)
        self.m_embedding_sum = None
        self.m_feature_sum = None

    def __hash__(self):
        head_id = self[0] if self else -1
        return hash(head_id)

    def append(self, mention):
        super(MentionCluster, self).append(mention)

        if mention and mention.embedding is not None:
            if self.m_embedding_sum is not None:
                self.m_embedding_sum += mention.embedding
            else:
                self.m_embedding_sum = np.array(mention.embedding)

        if mention and mention.feature is not None:
            if self.m_feature_sum is not None:
                self.m_feature_sum += mention.feature
            else:
                self.m_feature_sum = np.array(mention.feature)

    def get_avg_mention_embedding(self):
        return self.m_embedding_sum / len(self) \
            if self.m_embedding_sum is not None else None

    def get_avg_mention_feature(self):
        return self.m_feature_sum / len(self) \
            if self.m_feature_sum is not None else None
