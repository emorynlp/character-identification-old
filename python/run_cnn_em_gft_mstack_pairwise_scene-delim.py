from utils.readers import *
from utils.timer import Timer
from utils.transcritps import *
from components.features import *
from utils.evaluators import BCubeEvaluator
from gensim.models.keyedvectors import KeyedVectors
from components.oracles import SequentialStaticOracle
from models.cnn_em_gft import EntityMentionGroupFeatCNN_MStackPairwise

# Paths ####################################################
data_in = [
    # ("../data/friends-s1.feats.xs.json", [1], [2]),
    ("../data/friends-s1.feats.sm.json", range(1, 6), [7]),
    # ("../data/friends-s1.feats.json", range(1, 20), range(20, 22)),
    # ("../data/friends-s2.feats.json", filter(lambda x: x != 3, range(1, 22)), [22])
]

word2gender_path = "../data/gender.data"
word2vec_path = "../data/wiki_nyt_w2v_50d.bin"

# Parameters ##############################################
nb_m4entity = 5
nb_filters = 80

gpu_id = -1
evalOnly = False
nb_epoch = 200
eval_every = 10
batch_size = 128

utid = int(Timer.now()) % 1e6
model_out = "../learned_models/mm-cnn-mpair.1+2+3r.f%d.d50.f1+2.%d.m" % (nb_filters, utid)

nb_emb_feats = embdim = dftdim = None
###########################################################


def main():
    timer = Timer()

    #### Loading word2vec
    timer.start('load_word2vec')
    word2vec = KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
    print "Word2Vec data loaded w/ %d vocabularies of dimension %d - %.2fs" \
          % (len(word2vec.vocab), word2vec.syn0.shape[1], timer.end('load_word2vec'))

    #### Loading gender data
    timer.start('load_word2gender')
    word2gender = GenderDataReader.load(word2gender_path, True, True)
    print "Gender data loaded w/ %d vocabularies - %.2fs\n" % (len(word2gender), timer.end('load_word2gender'))

    #### Loading transcripts
    timer.start('load_transcript')
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

                key = "e%dsc%d" % (eid, scid)
                target = d_trn if eid in d_in[1] else d_dev if eid in d_in[2] else d_tst
                if key not in target:
                    target[key] = []
                target[key].append(m)

            m_trn += d_trn.values()
            m_dev += d_dev.values()
            m_tst += d_tst.values()

            print "Transcript loaded: %s w/ %d mentions" % (d_in[0], len(mentions))

    mc_trn, mc_dev, mc_tst = sum(map(lambda d: len(d), m_trn)), sum(map(lambda d: len(d), m_dev)), sum(map(lambda d: len(d), m_tst))
    print "%d transcript(s) loaded with %d speakers and %d documents (%d mentions). (Trn/Dev/Tst: %d(%d)/%d(%d)/%d(%d)) - %.2fs\n" \
          % (len(data_in), len(speakers), len(m_trn)+len(m_dev)+len(m_tst), len(m_all), len(m_trn),
             mc_trn, len(m_dev), mc_dev, len(m_tst), mc_tst, timer.end('load_transcript'))

    #### Extracting mention features
    timer.start('feature_extraction')
    m_all[0].id, m_all[-1].id = 0, len(m_all) - 1
    for m_idx, m in enumerate(m_all[1:-1], 1):
        m.id, m.m_prev, m.m_next = m_idx, m_all[m_idx-1], m_all[m_idx+1]
    m_extractor = MentionFeatureExtractor(word2vec, word2gender, speakers, pos_tags, dep_labels, ner_tags)
    for m in m_all:
        m.embedding, m.feature = m_extractor.extract(m)
    print "Vectorized and Extracted features from %d mentions. - %.2fs" % (len(m_all), timer.end('feature_extraction'))

    # Collection feature shape information
    nb_embs, embdim, embftdim, dftdim = \
        [len(e) for e in m_all[0].embedding], word2vec.syn0.shape[1], \
        len(m_all[0].feature), MentionPairFeatureExtractor().dftdim

    if not evalOnly:
        timer.start('model_training', 'batch_construction')
        #### Constructing trn/dev instances and clusters
        Xtrn, Ytrn, Ctrn = construct_instance_batch(m_trn, nb_m4entity, nb_embs, embdim, embftdim)
        print "Trn: %d instances of %d mentions from %d documents." % (len(Ytrn), mc_trn, len(Ctrn))
        Xdev, Ydev, Cdev = construct_instance_batch(m_dev, nb_m4entity, nb_embs, embdim, embftdim)
        print "Dev: %d instances of %d mentions from %d documents." % (len(Ydev), mc_dev, len(Cdev))
        print "Constructed Trn/Dev instances from static oracle - %.2fs\n" % timer.end('batch_construction')

        ### Model training and selection
        print "Initializing and Training model..."
        model = EntityMentionGroupFeatCNN_MStackPairwise(nb_m4entity, nb_embs, embdim, embftdim, dftdim, nb_filters, gpu_id=gpu_id)
        model.fit(m_trn, m_dev, Ctrn, Cdev, Xtrn, Ytrn, Xdev, Ydev,
                  eval_every=eval_every, batch_size=batch_size, nb_epoch=nb_epoch, model_out=model_out)
        print "Total training time: %.2fs\n" % timer.end('model_training')
    else:
        model = EntityMentionGroupFeatCNN_MStackPairwise(nb_m4entity, nb_embs, embdim, embftdim, dftdim, nb_filters, gpu_id=gpu_id)
        model.load_model(model_out)

    #### Evaluate trained model on test
    Xtst, Ytst, tst_cluster_docs_gold = construct_instance_batch(m_tst, nb_m4entity, nb_embs, embdim, embftdim)
    print "Evaluating model performance on test set..."

    tst_cluster_docs_pred = model.decode(m_tst)
    tst_p, tst_r, tst_f1 = BCubeEvaluator().evaluate_docs(tst_cluster_docs_gold, tst_cluster_docs_pred)
    print 'Evaluation - Tst: P/R/F: %.4f/%.4f/%.4f\n' % (tst_p, tst_r, tst_f1)


def construct_instance_batch(mention_docs, nb_m4entity, nb_embs, embdim, embftdim):
    nb_emb, em_ebft_padding = len(nb_embs), np.zeros(embftdim)
    em_embs_padding = [np.zeros((1, nb_emb, embdim)) for nb_emb in nb_embs]

    probs, cluster_docs = [], []
    m_gembs, m_ebfts = [[] for _ in xrange(nb_emb)], []
    em_gembs, em_ebfts = [[] for _ in xrange(nb_m4entity * nb_emb)], [[] for _ in xrange(nb_m4entity)]
    for mentions in mention_docs:
        oracle = SequentialStaticOracle(mentions)
        cluster_docs.append(oracle.all_clusters())

        m_m2feats = dict()
        for m in mentions:
            embs = [eb.reshape(1, neb, embdim) for eb, neb in zip(m.embedding, nb_embs)]
            m_m2feats[m] = (embs, m.feature.reshape(embftdim))

        m, c_pos, cs_neg = oracle.next()
        while m is not None:
            m_embs, m_ebft = m_m2feats[m]

            if c_pos:
                probs.append(1.0)
                for i, m_emb in enumerate(m_embs):
                    m_gembs[i] += [m_emb]
                m_ebfts += [m_ebft]

                cms = c_pos[-nb_m4entity:] if len(c_pos) >= nb_m4entity else [None] * (nb_m4entity - len(c_pos)) + c_pos
                cms.reverse()
                for i, cm in enumerate(cms):
                    em_embs, em_ebft = m_m2feats[cm] if cm in m_m2feats else (em_embs_padding, em_ebft_padding)

                    em_ebfts[i].append(em_ebft)
                    for j, em_emb in enumerate(em_embs):
                        em_gembs[i * nb_emb + j].append(em_emb)

            for c in cs_neg:
                probs.append(0.0)
                for i, m_emb in enumerate(m_embs):
                    m_gembs[i] += [m_emb]
                m_ebfts += [m_ebft]

                cms = c[-nb_m4entity:] if len(c) >= nb_m4entity else [None] * (nb_m4entity - len(c)) + c

                cms.reverse()
                for i, cm in enumerate(cms):
                    em_embs, em_ebft = m_m2feats[cm] if cm in m_m2feats else (em_embs_padding, em_ebft_padding)

                    em_ebfts[i].append(em_ebft)
                    for j, em_emb in enumerate(em_embs):
                        em_gembs[i * nb_emb + j].append(em_emb)

            m, c_pos, cs_neg = oracle.next()

    em_gembs = map(lambda x: np.array(x).astype('float32'), em_gembs)
    em_ebfts = map(lambda x: np.array(x).astype('float32'), em_ebfts)
    m_gembs = map(lambda x: np.array(x).astype('float32'), m_gembs)

    batch = em_gembs + em_ebfts + m_gembs + [np.array(m_ebfts)]
    pos_count = np.count_nonzero(probs)
    neg_count = len(probs) - pos_count

    print "#documents: %d, (+): %d, (-): %d, %s" % (len(cluster_docs), pos_count, neg_count, str(map(lambda a: a.shape, batch)))
    return batch, np.array(probs).astype('float32'), cluster_docs


if __name__ == "__main__":
    main()
