from utils.readers import *
from utils.transcritps import *
from components.features import *
from models.cnn_mm import MentionMentionCNN
from components.oracles import SequentialStaticOracle

# Paths ####################################################
data_in = [
    ("../data/friends-s1.feats.sm.json", range(1, 7), [7]),
    # ("../data/friends-s1.feats.json", range(1, 20), range(20, 22)),
    # ("../data/friends-s2.feats.json", filter(lambda x: x != 3, range(1, 22)), [22])
]

word2vec_path = "../data/wiki_nyt_w2v_50d.bin"
word2gender_path = "../data/gender.data"

# Parameters ##############################################
emb_nb_filters = 50
dft_nb_filters = 20
nb_m_prev = 10

nb_epoch = 100
eval_every = 10
batch_size = 32
model_out = "../learned_models/cnn.m"

nb_emb_feats = embdim = dftdim = None
###########################################################


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
    for m in m_trn + m_dev:
        m.embedding, m.feature = m_extractor.extract(m)
    print "Vectorized and Extracted features from %d mentions.\n" % len(m_all)

    nb_emb_feats, embdim = m_all[0].embedding.shape
    dftdim = len(m_all[0].feature)

    print "Constructing Trn/Dev instances from static oracle..."
    Xtrn, Ytrn, Ctrn = construct_instance_batch(m_trn, nb_emb_feats, embdim, dftdim, nb_m_prev)
    print "Trn: %d instances from %d mentions and %d gold clusters." % (len(Ytrn), len(m_trn), len(Ctrn))
    Xdev, Ydev, Cdev = construct_instance_batch(m_dev, nb_emb_feats, embdim, dftdim, nb_m_prev)
    print "Dev: %d instances from %d mentions and %d gold clusters.\n" % (len(Ydev), len(m_dev), len(Cdev))

    print "Initializing and Training model..."
    model = MentionMentionCNN(nb_emb_feats, embdim, dftdim, emb_nb_filters, dft_nb_filters)
    model.fit(m_trn, m_dev, Ctrn, Cdev, Xtrn, Ytrn, Xdev, Ydev, nb_m_prev=nb_m_prev,
              eval_every=eval_every, batch_size=batch_size, nb_epoch=nb_epoch, model_out=model_out)


def construct_instance_batch(mentions, nb_emb_feats, embdim, dftdim, nb_m_prev):
    oracle = SequentialStaticOracle(mentions)
    oracle.next()   # Read off the first mentions

    mention, pos_cluster, _ = oracle.next()
    m1_embs, m2_embs, m1_dfts, m2_dfts, probs = [], [], [], [], []
    while mention is not None:
        curr_dft = mention.feature.reshape(dftdim)
        curr_emb = mention.embedding.reshape(1, nb_emb_feats, embdim)

        pos_mentions = pos_cluster[-nb_m_prev:] if len(pos_cluster) > nb_m_prev else pos_cluster
        for pos_mention in pos_mentions:
            m1_embs.append(pos_mention.embedding.reshape(1, nb_emb_feats, embdim))
            m1_dfts.append(pos_mention.feature.reshape(dftdim))
            m2_embs.append(curr_emb)
            m2_dfts.append(curr_dft)
            probs.append(1.0)

        end_mid = mentions.index(mention)
        start_mid = max(0, end_mid - nb_m_prev)
        for prev in mentions[start_mid: end_mid]:
            if prev != pos_mention:
                m1_embs.append(prev.embedding.reshape(1, nb_emb_feats, embdim))
                m1_dfts.append(prev.feature.reshape(dftdim))
                m2_embs.append(curr_emb)
                m2_dfts.append(curr_dft)
                probs.append(0.0)

        mention, pos_cluster, _ = oracle.next()

    m1_embs, m2_embs = np.array(m1_embs), np.array(m2_embs)
    m1_dfts, m2_dfts = np.array(m1_dfts), np.array(m2_dfts)

    pos_count = np.count_nonzero(probs)
    neg_count = len(probs) - pos_count
    print "%s, (+): %d, (-): %d" % (str([m1_embs.shape, m2_embs.shape, m1_dfts.shape, m2_dfts.shape]), pos_count, neg_count)
    return [m1_embs, m2_embs, m1_dfts, m2_dfts], np.array(probs).reshape(len(probs), 1), oracle.all_clusters()

if __name__ == "__main__":
    main()
