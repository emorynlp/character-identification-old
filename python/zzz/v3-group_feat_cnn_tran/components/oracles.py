from abc import *
from structures.collections import MentionCluster


###########################################################
class AbstractOracle(object):

    @abstractmethod
    def initialize(self, mentions):
        return

    @abstractmethod
    def next(self):
        return

    @staticmethod
    def extract_labels(mentions):
        m2labels = dict()
        for mention in mentions:
            m2labels[mention] = mention.referent
            mention.referent = None
        return m2labels


###########################################################
class SequentialStaticOracle(AbstractOracle):
    def __init__(self, mentions):
        self.m_idx = self.mentions = None
        self.m2labels = self.m2clusters = None
        self.curr_mentions = self.curr_clusters = None

        self.initialize(mentions)

    def initialize(self, mentions):
        self.m_idx = 0
        self.mentions = mentions

        self.curr_mentions = set()
        self.curr_clusters = dict()

        self.m2labels = self.extract_labels(mentions)

    def mention_to_cluster_map(self):
        m_l2cluster = dict()
        for label in set(self.m2labels.values()):
            m_l2cluster[label] = MentionCluster()

        m_m2cluster = dict()
        for mention in self.m2labels.keys():
            cluster = m_l2cluster[self.m2labels[mention]]
            cluster.append(mention)

            m_m2cluster[mention] = cluster
        return m_m2cluster

    def all_clusters(self):
        m_l2cluster = dict()
        for label in set(self.m2labels.values()):
            m_l2cluster[label] = MentionCluster()

        for mention in self.m2labels.keys():
            m_l2cluster[self.m2labels[mention]].append(mention)
        return m_l2cluster.values()

    def next(self):
        if self.m_idx < len(self.mentions):
            mention = self.mentions[self.m_idx]
            self.m_idx += 1

            label = self.m2labels[mention]
            if label not in self.curr_clusters.keys():
                self.curr_clusters[label] = MentionCluster()

            pos_cluster = self.curr_clusters[label]
            neg_clusters = self.curr_clusters.values()
            neg_clusters.remove(pos_cluster)

            updated_cluster = MentionCluster(pos_cluster)
            updated_cluster.append(mention)
            self.curr_clusters[label] = updated_cluster

            return mention, pos_cluster, neg_clusters

        return None, None, None
