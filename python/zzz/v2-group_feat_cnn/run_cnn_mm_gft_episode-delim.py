from time import time
from utils.readers import *
from utils.transcritps import *
from components.features import *
from utils.evaluators import BCubeEvaluator
from models.cnn_mm_gft import MentionMentionCNN
from components.oracles import SequentialStaticOracle

# Paths ####################################################
data_in = [
    ("../data/friends-s1.feats.sm.json", range(1, 6), [7]),
    # ("../data/friends-s1.feats.json", range(1, 20), range(20, 22)),
    # ("../data/friends-s2.feats.json", filter(lambda x: x != 3, range(1, 22)), [22])
]

word2gender_path = "../data/gender.data"
word2vec_path = "../data/wiki_nyt_w2v_50d.bin"

# Parameters ##############################################
nb_filters = 80

gpu_id = -1
evalOnly = False
nb_epoch = 100
eval_every = 10
batch_size = 128

utid = int(time()) % 1e6
model_out = "../learned_models/mm-cnn.1+2+3r.f%d.d50.f1+2.%d.m" % (nb_filters, utid)

nb_emb_feats = embdim = dftdim = None
###########################################################


def main():
    #### Loading word2vec
    with open(word2vec_path, 'r') as f:
        word2vec = Word2VecReader.load_bin(f)
        print "Word2Vec data loaded w/ %d vocabularies" % len(word2vec)

    #### Loading gender data
    with open(word2gender_path, 'r') as f:
        word2gender = GenderDataReader.load(f, True, True)
        print "Gender data loaded w/ %d vocabularies\n" % len(word2gender)

    #### Loading transcripts
    m_all, m_trn, m_dev, m_tst = ([], [], [], [])
    speakers, pos_tags, dep_labels, ner_tags = (set(), set(), set(), set())
    for d_in in data_in:
        with open(d_in[0], 'r') as f:
            season, _, mentions = TranscriptJsonReader.read_season(f)

            speakers = speakers.union(TranscriptUtils.collect_speakers(season, False))
            pos_tags = pos_tags.union(TranscriptUtils.collect_pos_tags(season))
            dep_labels = dep_labels.union(TranscriptUtils.collect_dep_labels(season))
            ner_tags = ner_tags.union(TranscriptUtils.collect_ner_tags(season))

            m_all += mentions
            d_trn, d_dev, d_tst = dict(), dict(), dict()
            for m in mentions:
                scene = m.tokens[0].parent_scene()
                episode = scene.parent_episode()
                scid, eid = scene.id, episode.id

                key = "e%d" % eid
                target = d_trn if eid in d_in[1] else d_dev if eid in d_in[2] else d_tst
                if key not in target:
                    target[key] = []
                target[key].append(m)

            m_trn += d_trn.values()
            m_dev += d_dev.values()
            m_tst += d_tst.values()

            print "Transcript loaded: %s w/ %d mentions" % (d_in[0], len(mentions))
    mc_trn, mc_dev, mc_tst = sum(map(lambda d: len(d), m_trn)), sum(map(lambda d: len(d), m_dev)), sum(map(lambda d: len(d), m_tst))
    print "%d transcript(s) loaded with %d speakers and %d documents (%d mentions). (Trn/Dev/Tst: %d(%d)/%d(%d)/%d(%d))" \
          % (len(data_in), len(speakers), len(m_trn)+len(m_dev)+len(m_tst), len(m_all), len(m_trn), mc_trn, len(m_dev), mc_dev, len(m_tst), mc_tst)

    #### Extracting mention features (Multi-threaded)
    m_all[0].id, m_all[-1].id, start_time = 0, len(m_all) - 1, time()
    for m_idx, m in enumerate(m_all[1:-1], 1):
        m.id, m.m_prev, m.m_next = m_idx, m_all[m_idx-1], m_all[m_idx+1]
    m_extractor = MentionFeatureExtractor(word2vec, word2gender, speakers, pos_tags, dep_labels, ner_tags)
    for m in m_all:
        m.embedding, m.feature = m_extractor.extract(m)
    print "Vectorized and Extracted features from %d mentions. - %.2fs\n" % (len(m_all), time() - start_time)

    # Collection feature shape information
    nb_embs, embdim, embftdim = [len(e) for e in m_all[0].embedding], len(word2vec.values()[0]), len(m_all[0].feature)

    if not evalOnly:
        #### Constructing trn/dev instances and clusters
        print "Constructing Trn/Dev instances from static oracle..."
        Xtrn, Ytrn, Ctrn = construct_instance_batch(m_trn, nb_embs, embdim, embftdim)
        print "Trn: %d instances from %d mentions of %d documents." % (len(Ytrn), mc_trn, len(Ctrn))
        Xdev, Ydev, Cdev = construct_instance_batch(m_dev, nb_embs, embdim, embftdim)
        print "Dev: %d instances from %d mentions of %d documents.\n" % (len(Ydev), mc_dev, len(Cdev))

        #### Model training and selection
        print "Initializing and Training model..."
        model = MentionMentionCNN(nb_embs, embdim, embftdim, len(Xtrn[-1][0]), nb_filters, gpu_id=gpu_id)
        model.fit(m_trn, m_dev, Ctrn, Cdev, Xtrn, Ytrn, Xdev, Ydev,
                  eval_every=eval_every, batch_size=batch_size, nb_epoch=nb_epoch, model_out=model_out)
        print ""

    #### Evaluate trained model on test
    Xtst, Ytst, tst_cluster_docs_gold = construct_instance_batch(m_tst, nb_embs, embdim, embftdim)
    model = MentionMentionCNN(nb_embs, embdim, embftdim, len(Xtst[-1][0]), nb_filters, gpu_id=gpu_id)
    model.load_model(model_out)
    print "Evaluating model performance on test set..."
    tst_cluster_docs_pred = model.decode(m_tst)
    tst_p, tst_r, tst_f1 = BCubeEvaluator().evaluate_docs(tst_cluster_docs_gold, tst_cluster_docs_pred)
    print 'Evaluation - Tst: P/R/F: %.4f/%.4f/%.4f\n' % (tst_p, tst_r, tst_f1)


def construct_instance_batch(mention_docs, nb_embs, embdim, embftdim):
    mm_extractor = MentionPairFeatureExtractor()
    m1_embs, m2_embs, m1_ebfts, m2_ebfts, mm_dfts, probs, cluster_docs = [], [], [], [], [], [], []
    for i in range(len(nb_embs)):
        m1_embs.append([])
        m2_embs.append([])

    for mentions in mention_docs:
        oracle = SequentialStaticOracle(mentions)
        m_m2cluster = oracle.mention_to_cluster_map()

        cluster_docs.append(list(set(m_m2cluster.values())))
        for c_idx, curr in enumerate(mentions[1:], 1):
            pos_cluster = m_m2cluster[curr]

            curr_ebft = curr.feature.reshape(embftdim)
            curr_embs = [eb.reshape(1, neb, embdim) for eb, neb in zip(curr.embedding, nb_embs)]

            for prev in reversed(mentions[:c_idx]):
                prev_ebft = prev.feature.reshape(embftdim)
                prev_embs = [eb.reshape(1, neb, embdim) for eb, neb in zip(prev.embedding, nb_embs)]

                for idx, prev_emb, curr_emb in zip(range(len(nb_embs)), prev_embs, curr_embs):
                    m1_embs[idx].append(prev_emb)
                    m2_embs[idx].append(curr_emb)
                m1_ebfts.append(prev_ebft)
                m2_ebfts.append(curr_ebft)

                mm_dft = mm_extractor.extract((prev, curr))
                mm_dfts.append(mm_dft)

                if prev in pos_cluster:
                    probs.append(1.0)
                    break
                probs.append(0.0)

    m1_embs, m1_ebfts = map(lambda e: np.array(e), m1_embs), np.array(m1_ebfts)
    m2_embs, m2_ebfts = map(lambda e: np.array(e), m2_embs), np.array(m2_ebfts)
    mm_dfts = np.array(mm_dfts)

    pos_count = np.count_nonzero(probs)
    neg_count = len(probs) - pos_count

    shape_str = str(map(lambda a: a.shape, m1_embs + m2_embs + [m1_ebfts, m2_ebfts, mm_dfts]))
    print "#documents: %d, (+): %d, (-): %d, %s" % (len(cluster_docs), pos_count, neg_count, shape_str)
    return m1_embs + m2_embs + [m1_ebfts, m2_ebfts, mm_dfts], np.array(probs).astype('float32'), cluster_docs


if __name__ == "__main__":
    main()
