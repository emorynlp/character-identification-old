import random
from utils.readers import *
from utils.transcritps import *
from components.features import *
from models.cnn import EntityMentionCNN
from components.oracles import SequentialStaticOracle

# Paths ####################################################
data_in = [
    # ("../data/friends-s1.feats.sm.json", [1], [2]),
    ("../data/friends-s1.feats.json", range(1, 20), range(20, 22)),
    # ("../data/friends-s2.feats.json", filter(lambda x: x != 3, range(1, 22)), [22])
]

word2vec_path = "../data/wiki_nyt_w2v_50d.bin"
word2gender_path = "../data/gender.data"

# Parameters ##############################################
neg_sample = 0.01

conv_rows = 1
entity_chns = 5
emb_nb_filters = 20

nb_epoch = 100
model_out = "../learned_models/cnn.m"


###########################################################
nb_emb_feats = embdim = dftdim = None
def main():
    with open(word2vec_path, 'r') as f:
        word2vec = Word2VecReader.load_bin(f)
        print "Word2Vec data loaded w/ %d vocabularies" % len(word2vec)

    with open(word2gender_path, 'r') as f:
        word2gender = GenderDataReader.load(f, True, True)
        print "Gender data loaded w/ %d vocabularies\n" % len(word2gender)

    m_all, m_trn, m_dev, m_tst = ([], [], [], [])
    speakers, pos_tags, dep_labels = (set(), set(), set())
    for d_in in data_in:
        with open(d_in[0], 'r') as f:
            season, _, mentions = TranscriptJsonReader.read_season(f)

            speakers = speakers.union(TranscriptUtils.collect_speakers(season, False))
            pos_tags = pos_tags.union(TranscriptUtils.collect_pos_tags(season))
            dep_labels = dep_labels.union(TranscriptUtils.collect_dep_labels(season))

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
          % (len(data_in), len(speakers), len(m_all), len(m_trn), len(m_dev), len(m_tst))

    m_extractor = MentionFeatureExtractor(word2vec, word2gender, speakers, pos_tags, dep_labels)
    for m in m_all:
        m.embedding, m.feature = m_extractor.extract(m)
    print "Vectorized and Extracted features from %d mentions.\n" % len(m_all)

    nb_emb_feats, embdim = m_all[0].embedding.shape
    dftdim = len(m_all[0].feature)

    print "Constructing Trn/Dev instances from static oracle..."
    Xtrn, Ytrn, Ctrn = construct_instance_batch(m_trn, neg_sample, nb_emb_feats, embdim, dftdim)
    print "Trn: %d instances from %d mentions and %d gold clusters." % (len(Ytrn), len(m_trn), len(Ctrn))
    Xdev, Ydev, Cdev = construct_instance_batch(m_dev, neg_sample, nb_emb_feats, embdim, dftdim)
    print "Dev: %d instances from %d mentions and %d gold clusters.\n" % (len(Ydev), len(m_dev), len(Cdev))

    print "Initializing and Training model..."
    model = EntityMentionCNN(entity_chns, nb_emb_feats, embdim, dftdim, conv_rows, emb_nb_filters)
    model.fit(m_trn, m_dev, Ctrn, Cdev, Xtrn, Ytrn, Xdev, Ydev, nb_epoch=1, model_out=model_out)


def construct_instance_batch(mentions, neg_sample, nb_emb_feats, embdim, dftdim):
    oracle = SequentialStaticOracle(mentions)
    e_extractor = EntityFeatureExtractor(mentions[0].embedding.shape, mentions[0].feature.shape)
    e_embeddings, m_embeddings, e_features, m_features, probs = ([], [], [], [], [])

    mention, pos_cluster, neg_clusters = oracle.next()
    while mention is not None:
        m_embedding = mention.embedding.reshape(1, nb_emb_feats, embdim)
        m_feature = mention.feature

        e_embedding, e_feature = e_extractor.extract(pos_cluster, True, entity_chns)
        e_embeddings.append(e_embedding)
        e_features.append(e_feature.reshape(1, entity_chns, dftdim))

        m_embeddings.append(m_embedding)
        m_features.append(m_feature)
        probs.append(1.0)

        # Negative sampling by ratio
        ns_size = int(round(len(neg_clusters) * neg_sample))
        indices = set()
        while len(indices) < ns_size:
            id = random.randint(0, len(neg_clusters)-1)

            if id not in indices:
                indices.add(id)

                e_embedding, e_feature = e_extractor.extract(pos_cluster, True, entity_chns)
                e_embeddings.append(e_embedding)
                e_features.append(e_feature.reshape(1, entity_chns, dftdim))

                m_embeddings.append(m_embedding)
                m_features.append(m_feature)

                probs.append(0.0)

        mention, pos_cluster, neg_clusters = oracle.next()

    e_embeddings = np.swapaxes(e_embeddings, 0, 1)
    e_embeddings = [es.reshape(len(probs), 1, nb_emb_feats, embdim) for es in e_embeddings]

    instances = e_embeddings + [np.array(m_embeddings), np.array(e_features), np.array(m_features)]
    return instances, np.array(probs).reshape(len(probs), 1), oracle.all_clusters()

if __name__ == "__main__":
    main()
