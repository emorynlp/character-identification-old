from time import time
from utils.readers import *
from utils.transcritps import *
from multiprocessing import Pool
from components.features import *
from models.cnn_mm import MentionMentionCNN
from components.oracles import SequentialStaticOracle
from utils.multiprocessing import MultiprocessingUtils

# Paths ####################################################
data_in = [
    # ("../data/friends-s1.feats.sm.json", range(1, 7), [7]),
    ("../data/friends-s1.feats.json", range(1, 20), range(20, 22)),
    # ("../data/friends-s2.feats.json", filter(lambda x: x != 3, range(1, 22)), [22])
]

word2vec_path = "../data/wiki_nyt_w2v_50d.bin"
word2gender_path = "../data/gender.data"

# Parameters ##############################################
emb_nb_filters = 40
dft_nb_filters = 20

thread_count = 4
nb_epoch = 100
eval_every = 10
batch_size = 32
model_out = "../learned_models/cnn.m"

nb_emb_feats = embdim = dftdim = None
###########################################################


def MentionExtractionFn(extractor, mentions):
    extracted = []
    for mention in mentions:
        mention.embedding, mention.feature = extractor.extract(mention)
        extracted.append(mention)
    return extracted


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
    print "%d transcript(s) loaded with %d speakers and %d documents (%d mentions). (Trn/Dev/Tst: %d(%d)/%d(%d)/%d(%d))" \
          % (len(data_in), len(speakers), len(m_trn)+len(m_dev)+len(m_tst), len(m_all), len(m_trn), mc_trn, len(m_dev), mc_dev, len(m_tst), mc_tst)

    #### Extracting mention features (Multi-threaded)
    m_extractor = MentionFeatureExtractor(word2vec, word2gender, speakers, pos_tags, dep_labels, ner_tags)
    indices, pool = MultiprocessingUtils.get_array_split_indices(len(m_all), thread_count), Pool(thread_count)

    start_time, processes = time(), []
    for idx in xrange(0, thread_count):
        mentions = m_all[indices[idx]:indices[idx+1]]
        processes.append(pool.apply_async(MentionExtractionFn, (m_extractor, mentions)))
    pool.close()
    pool.join()

    for idx, m, e in zip(range(len(m_all)), m_all, sum([p.get() for p in processes], [])):
        m.id, m.embedding, m.feature = idx, e.embedding, e.feature
    print "Vectorized and Extracted features from %d mentions. - %ds\n" % (len(m_all), time()-start_time)

    #### Constructing trn/dev instances and clusters
    nb_emb_feats, embdim = m_all[0].embedding.shape
    dftdim = len(m_all[0].feature)
    print "Constructing Trn/Dev instances from static oracle..."
    Xtrn, Ytrn, Ctrn = construct_instance_batch(m_trn, nb_emb_feats, embdim, dftdim)
    print "Trn: %d instances from %d mentions of %d documents." % (len(Ytrn), mc_trn, len(Ctrn))
    Xdev, Ydev, Cdev = construct_instance_batch(m_dev, nb_emb_feats, embdim, dftdim)
    print "Dev: %d instances from %d mentions of %d documents.\n" % (len(Ydev), mc_dev, len(Cdev))

    #### Model training and selection
    print "Initializing and Training model..."
    model = MentionMentionCNN(nb_emb_feats, embdim, dftdim, emb_nb_filters, dft_nb_filters)
    model.fit(m_trn, m_dev, Ctrn, Cdev, Xtrn, Ytrn, Xdev, Ydev,
              eval_every=eval_every, batch_size=batch_size, nb_epoch=nb_epoch, model_out=model_out)


def construct_instance_batch(mention_docs, nb_emb_feats, embdim, dftdim):
    m1_embs, m2_embs, m1_dfts, m2_dfts, probs, cluster_docs = [], [], [], [], [], []
    for mentions in mention_docs:
        oracle = SequentialStaticOracle(mentions)
        m_m2cluster = oracle.mention_to_cluster_map()

        cluster_docs.append(list(set(m_m2cluster.values())))
        for c_idx, curr in enumerate(mentions[1:], 1):
            pos_cluster = m_m2cluster[curr]
            curr_dft = curr.feature.reshape(dftdim)
            curr_emb = curr.embedding.reshape(1, nb_emb_feats, embdim)

            for prev in reversed(mentions[:c_idx]):
                prev_dft = prev.feature.reshape(dftdim)
                prev_emb = prev.embedding.reshape(1, nb_emb_feats, embdim)

                m1_embs.append(prev_emb)
                m2_embs.append(curr_emb)
                m1_dfts.append(prev_dft)
                m2_dfts.append(curr_dft)

                if prev in pos_cluster:
                    probs.append(1.0)
                    break
                probs.append(0.0)

    m1_embs, m2_embs = np.array(m1_embs), np.array(m2_embs)
    m1_dfts, m2_dfts = np.array(m1_dfts), np.array(m2_dfts)

    pos_count = np.count_nonzero(probs)
    neg_count = len(probs) - pos_count

    print "%s, #documents: %d, (+): %d, (-): %d" % \
          (str([m1_embs.shape, m2_embs.shape, m1_dfts.shape, m2_dfts.shape]), len(cluster_docs), pos_count, neg_count)
    return [m1_embs, m2_embs, m1_dfts, m2_dfts], np.array(probs).reshape(len(probs), 1), cluster_docs


if __name__ == "__main__":
    main()
