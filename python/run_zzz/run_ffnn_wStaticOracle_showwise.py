import random
from models.ffnn import *
from utils.readers import *
from utils.transcritps import *
from components.features import *
from components.oracles import SequentialStaticOracle

# Paths ####################################################
data_in = [
    ("../data/friends-s1.feats.json", range(1, 20), range(20, 22)),
    # ("../data/friends-s2.feats.json", filter(lambda x: x != 3, range(1, 22)), [22])
]

word2vec_path = "../data/wiki_nyt_w2v_50d.bin"
word2gender_path = "../data/gender.data"

# Parameters ##############################################
neg_sample = 0.01

nb_epoch = 20
hnode_embd = 50
hnode_feat = 5

model_out = "../learned_models/ffnn.m"


###########################################################
def main():
    with open(word2vec_path, 'r') as f:
        word2vec = Word2VecReader.load_bin(f)
        print "Word2Vec data loaded w/ %d vocabularies" % len(word2vec)

    with open(word2gender_path, 'r') as f:
        word2gender = GenderDataReader.load(f, True, True)
        print "Gender data loaded w/ %d vocabularies\n" % len(word2gender)

    spk2idx_map = dict()
    m_all, m_trn, m_dev, m_tst = ([], [], [], [])
    for d_in in data_in:
        with open(d_in[0], 'r') as f:
            season, _, mentions = TranscriptJsonReader.read_season(f)
            spk2idx_map = TranscriptUtils.create_speaker2idx_map(season, spk2idx_map)

            m_all += mentions
            for m in mentions:
                episode = m.tokens[0].parent_scene().parent_episode()
                eid = episode.id

                target = m_trn if eid in d_in[1] \
                    else m_dev if eid in d_in[2] \
                    else m_tst
                target.append(m)

            print "Transcript loaded: %s w/ %d mentions" % (d_in[0], len(mentions))

    print "%d transcript(s) loaded with %d speakers and %d mentions. (Trn/Dev/Tst: %d/%d/%d)" \
          % (len(data_in), len(spk2idx_map), len(m_all), len(m_trn), len(m_dev), len(m_tst))

    m_extractor = MentionFeatureExtractor(word2vec, word2gender, spk2idx_map)
    for m in m_all:
        m.embedding, m.feature = m_extractor.extract(m)
    print "Vectorized and Extracted features from %d mentions.\n" % len(m_all)

    print "Constructing Trn/Dev instances from static oracle..."
    Etrn_embds, Etrn_feats, Mtrn_embds, Mtrn_feats, Ytrn_gold, trn_clusters_gold = construct_instance_batch(m_trn, neg_sample)
    print "Trn: %d instances from %d mentions and %d gold clusters." % (len(Ytrn_gold), len(m_trn), len(trn_clusters_gold))
    Edev_embds, Edev_feats, Mdev_embds, Mdev_feats, Ydev_gold, dev_clusters_gold = construct_instance_batch(m_dev, 0.0)
    print "Dev: %d instances from %d mentions and %d gold clusters.\n" % (len(Ydev_gold), len(m_dev), len(dev_clusters_gold))

    print "Initializing and Training model..."
    model = FeedForwardModel(len(Mtrn_embds[0]), len(Mtrn_feats[0]), hnode_embd, hnode_feat)
    model.fit(m_trn, m_dev, trn_clusters_gold, dev_clusters_gold,
              Etrn_embds, Mtrn_embds, Etrn_feats, Mtrn_feats, Ytrn_gold,
              Edev_embds, Mdev_embds, Edev_feats, Mdev_feats, Ydev_gold,
              nb_epoch=nb_epoch, model_out=model_out)


def construct_instance_batch(mentions, neg_sample):
    oracle = SequentialStaticOracle(mentions)
    e_extractor = EntityFeatureExtractor(len(mentions[0].embedding), len(mentions[0].feature))
    e_embds, e_feats, m_embds, m_feats, probs = ([], [], [], [], [])

    mention, pos_cluster, neg_clusters = oracle.next()
    while mention is not None:
        e_embd, e_feat = e_extractor.extract(pos_cluster)

        e_embds.append(e_embd)
        e_feats.append(e_feat)
        m_embds.append(mention.embedding)
        m_feats.append(mention.feature)
        probs.append(1.0)

        # Negative sampling by ratio
        ns_size = int(round(len(neg_clusters) * neg_sample))
        indices = set()
        while len(indices) < ns_size:
            id = random.randint(0, len(neg_clusters)-1)

            if id not in indices:
                indices.add(id)
                e_embd, e_feat = e_extractor.extract(neg_clusters[id])

                e_embds.append(e_embd)
                e_feats.append(e_feat)
                m_embds.append(mention.embedding)
                m_feats.append(mention.feature)
                probs.append(0.0)

        mention, pos_cluster, neg_clusters = oracle.next()

    return np.array(e_embds), np.array(e_feats), np.array(m_embds), np.array(m_feats), \
           np.array(probs).reshape(len(probs), 1), oracle.all_clusters()

if __name__ == "__main__":
    main()
