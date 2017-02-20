from abc import *


###########################################################
class AbstractEvaluator(object):
    @abstractmethod
    def evaluate(self, gold_clusters, auto_clusters):
        return

    @staticmethod
    def f1_score(precision, recall):
        return 2 * precision * recall / (precision + recall)

    @staticmethod
    def create_mention2cluster_map(clusters):
        m = dict()
        for cluster in clusters:
            for mention in cluster:
                m[mention] = cluster

        return m


###########################################################
class BCubeEvaluator(AbstractEvaluator):
    def evaluate_docs(self, gold_cluster_docs, auto_cluster_docs):
        pc = rc = mc = 0
        for gold_clusters, auto_clusters in zip(gold_cluster_docs, auto_cluster_docs):
            gold_m2c_map = self.create_mention2cluster_map(gold_clusters)
            auto_m2c_map = self.create_mention2cluster_map(auto_clusters)
            mentions = auto_m2c_map.keys()

            mc += len(mentions)
            for mention in mentions:
                gold_cluster = gold_m2c_map.get(mention)
                auto_cluster = auto_m2c_map.get(mention)

                correct = len(set(gold_cluster).intersection(set(auto_cluster)))
                pc += float(correct) / len(auto_cluster)
                rc += float(correct) / len(gold_cluster)

        p = pc / mc
        r = rc / mc

        return p, r, self.f1_score(p, r)

    def evaluate_doc(self, gold_clusters, auto_clusters):
        gold_m2c_map = self.create_mention2cluster_map(gold_clusters)
        auto_m2c_map = self.create_mention2cluster_map(auto_clusters)
        mentions = auto_m2c_map.keys()

        pc = rc = 0
        for mention in mentions:
            gold_cluster = gold_m2c_map.get(mention)
            auto_cluster = auto_m2c_map.get(mention)

            correct = len(set(gold_cluster).intersection(set(auto_cluster)))
            pc += float(correct) / len(auto_cluster)
            rc += float(correct) / len(gold_cluster)

        p = pc / len(mentions)
        r = rc / len(mentions)

        return p, r, self.f1_score(p, r)
