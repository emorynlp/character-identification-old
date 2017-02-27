import os
import numpy as np
import tensorflow as tf
from utils.timer import Timer
from keras import backend as K
from sklearn.metrics import accuracy_score
from utils.evaluators import BCubeEvaluator
from structures.collections import MentionCluster
from keras.models import Model, save_model, load_model
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers import Input, Reshape, Flatten, Dense, Dropout, merge

tf.logging.set_verbosity(tf.logging.ERROR)
tf.python.control_flow_ops = tf


class EntityMentionGroupFeatCNNDecoder(object):

    @staticmethod
    def decode(model, mention_docs):
        nb_emb, nb_m4entity, cluster_docs = len(model.nb_embs), model.nb_m4entity, []
        em_embs_padding = [np.zeros((1, nb_emb, model.embdim)) for nb_emb in model.nb_embs]
        em_ebft_padding = np.zeros(model.embftdim)

        for mentions in mention_docs:
            m_m2feats, clusters = dict(), [MentionCluster([mentions[0]])]

            for m in mentions:
                embs = [eb.reshape(1, neb, model.embdim).astype('float32') for eb, neb in zip(m.embedding, model.nb_embs)]
                m_m2feats[m] = (embs, m.feature.reshape(model.embftdim).astype('float32'))

            for curr in mentions:
                curr_embs, curr_ebft = m_m2feats[curr]
                c_count, m_gembs, m_ebfts = len(clusters), [[] for _ in xrange(nb_emb)], []
                for i, m_emb in enumerate(curr_embs):
                    m_gembs[i] += [m_emb] * c_count
                m_ebfts += [curr_ebft] * c_count

                em_gembs, em_ebfts = [[] for _ in xrange(nb_m4entity * nb_emb)], [[] for _ in xrange(nb_m4entity)]
                for cluster in clusters:
                    cms = cluster[-nb_m4entity:] if len(cluster) >= nb_m4entity \
                        else [None] * (nb_m4entity-len(cluster)) + cluster
                    cms.reverse()

                    for i, cm in enumerate(cms):
                        cm_embs, cm_ebft = m_m2feats[cm] if cm in m_m2feats else (em_embs_padding, em_ebft_padding)
                        for j, cm_emb in enumerate(cm_embs):
                            em_gembs[i * nb_emb + j].append(cm_emb)
                        em_ebfts[i].append(cm_ebft)

                em_gembs = map(lambda x: np.array(x).astype('float32'), em_gembs)
                em_ebfts = map(lambda x: np.array(x).astype('float32'), em_ebfts)
                m_gembs = map(lambda x: np.array(x).astype('float32'), m_gembs)

                instances = em_gembs + em_ebfts + m_gembs + [np.array(m_ebfts)]
                predictions = model.predict(instances)

                if np.amax(predictions) > 0.5:
                    c_target = clusters[np.argmax(predictions)]
                else:
                    c_target = MentionCluster()
                    clusters.append(c_target)
                c_target.append(curr)

            cluster_docs.append(clusters)

        return cluster_docs


class AbstractEntityMentionGroupFeatCNN(object):
    def __init__(self):
        self.model = None

    def fit(self, mentions_trn, mentions_dev, trn_cluster_docs_gold, dev_cluster_docs_gold,
            Xtrn, Ytrn, Xdev, Ydev, eval_every=1, nb_epoch=20, batch_size=32, model_out=None):

        best_trn_scores, best_dev_scores, best_epoch, total_time = ([0]*3, [0]*3, 0, 0)
        decoder, evaluator = EntityMentionGroupFeatCNNDecoder(), BCubeEvaluator()

        timer = Timer()
        for e in range(nb_epoch/eval_every):
            timer.start('epoch', 'training_step')
            self.model.fit(Xtrn, Ytrn, batch_size=batch_size, nb_epoch=eval_every, shuffle=True, verbose=0)
            trn_time = timer.end('training_step')

            trn_accuracy = accuracy_score(Ytrn, np.round(self.predict(Xtrn)))
            dev_accuracy = accuracy_score(Ydev, np.round(self.predict(Xdev)))

            timer.start('decoding_step')
            trn_cluster_docs_pred = decoder.decode(self, mentions_trn)
            trn_p, trn_r, trn_f1 = evaluator.evaluate_docs(trn_cluster_docs_gold, trn_cluster_docs_pred)

            dev_cluster_docs_pred = decoder.decode(self, mentions_dev)
            dev_p, dev_r, dev_f1 = evaluator.evaluate_docs(dev_cluster_docs_gold, dev_cluster_docs_pred)
            decode_time = timer.end('decoding_step')

            if best_dev_scores[2] < dev_f1:
                best_epoch = e
                best_dev_scores = [dev_p, dev_r, dev_f1]
                best_trn_scores = [trn_p, trn_r, trn_f1]

                if model_out is not None:
                    self.save_model(model_out)

            lapse = timer.end('epoch')
            print 'Epoch %3d - Trn Accu(P/R/F): %.4f(%.4f/%.4f/%.4f), Dev Accu(P/R/F): %.4f(%.4f/%.4f/%.4f) - %4.2fs'\
                  % ((e+1)*eval_every, trn_accuracy, trn_p, trn_r, trn_f1, dev_accuracy, dev_p, dev_r, dev_f1, lapse)
            print '\tTime breakdown: Train %.2fs, Decode %.2fs' % (trn_time, decode_time)

        '\nTraining Summary:'
        print 'Best epoch: %d, Trn P/R/F: %.6f/%.6f/%.6f, Dev P/R/F : %.6f/%.6f/%.6f' % \
              ((best_epoch+1)*eval_every, best_trn_scores[0], best_trn_scores[1], best_trn_scores[2],
               best_dev_scores[0], best_dev_scores[1], best_dev_scores[2])

        if model_out is not None:
            print 'Model saved to %s' % model_out

    def decode(self, mention_docs):
        return EntityMentionGroupFeatCNNDecoder.decode(self, mention_docs)

    def predict(self, Xtst):
        return self.model.predict(Xtst)

    def load_model(self, file_path):
        try:
            self.model = load_model(file_path)
        except IOError:
            raise IOError("Can't load model file %s" % file_path)

    def save_model(self, file_path):
        save_model(self.model, file_path)


class EntityMentionGroupFeatCNN_MStackConvAll(AbstractEntityMentionGroupFeatCNN):
    def __init__(self, nb_m4entity, nb_embs, embdim, embftdim, dftdim, nb_filters, gpu_id=-1):
        super(EntityMentionGroupFeatCNN_MStackConvAll, self).__init__()
        self.nb_m4entity, self.nb_embs, self.embdim, self.embftdim, self.dftdim = \
            nb_m4entity, nb_embs, embdim, embftdim, dftdim

        if gpu_id >= 0:
            gpu_options = tf.GPUOptions(visible_device_list=str(gpu_id), allow_growth=True)
            K.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = "-1"

        with tf.device('/cpu:0' if gpu_id < 0 else '/gpu:0'):
            # Constant layers
            flatten, dropout = Flatten(), Dropout(0.8)

            # Input layers
            em_inp_ebfts = [Input(shape=(embftdim,)) for _ in range(nb_m4entity)]
            em_inp_gembs = [[Input(shape=(1, nb_emb, embdim)) for nb_emb in nb_embs] for _ in range(nb_m4entity)]

            m_inp_ebft = Input(shape=(embftdim,))
            m_inp_embs = [Input(shape=(1, nb_emb, embdim)) for nb_emb in nb_embs]

            ########################################################################
            # Mention representation model
            m_rows_reshape = Reshape((1, 3 * len(nb_embs), nb_filters))
            m_ftg_pool = MaxPooling2D(pool_size=(3 * len(nb_embs), 1))
            m_ftg_conv = Convolution2D(nb_filters, 1, nb_filters, activation='tanh')

            m_convs_1r, m_convs_2r, m_convs_3r, m_pools_1r, m_pools_2r, m_pools_3r = [[] for _ in xrange(6)]
            for nb_emb in nb_embs:
                m_convs_1r.append(Convolution2D(nb_filters, 1, embdim, activation='tanh'))
                m_convs_2r.append(Convolution2D(nb_filters, 2, embdim, activation='tanh'))
                m_convs_3r.append(Convolution2D(nb_filters, 3, embdim, activation='tanh'))
                m_pools_1r.append(MaxPooling2D(pool_size=(nb_emb - 0, 1)))
                m_pools_2r.append(MaxPooling2D(pool_size=(nb_emb - 1, 1)))
                m_pools_3r.append(MaxPooling2D(pool_size=(nb_emb - 2, 1)))

            def mention_repr(m_embs, m_ebft):
                gft1rs, gft2rs, gft3rs = [], [], []
                for inp_emb, m_conv_1r, m_conv_2r, m_conv_3r, m_pool_1r, m_pool_2r, m_pool_3r \
                        in zip(m_embs, m_convs_1r, m_convs_2r, m_convs_3r, m_pools_1r, m_pools_2r, m_pools_3r):
                    gft1rs.append(dropout(flatten(m_pool_1r(m_conv_1r(inp_emb)))))
                    gft2rs.append(dropout(flatten(m_pool_2r(m_conv_2r(inp_emb)))))
                    gft3rs.append(dropout(flatten(m_pool_3r(m_conv_3r(inp_emb)))))
                m_ftg_matrix = m_rows_reshape(merge(gft1rs + gft2rs + gft3rs, mode='concat'))
                m_emb_vec = flatten(m_ftg_pool(m_ftg_conv(m_ftg_matrix)))

                return merge([m_emb_vec, m_ebft], mode='concat')

            # Entity representation model
            em_rows_reshape = Reshape((1, nb_m4entity, nb_filters+embftdim))
            em_conv = Convolution2D(nb_filters+embftdim, 1, nb_filters+embftdim, activation='tanh')
            em_pool = MaxPooling2D(pool_size=(nb_m4entity, 1))

            def entity_repr(mention_reprs):
                em_matrix = em_rows_reshape(merge(mention_reprs, mode='concat')
                                            if len(mention_reprs) > 1 else mention_reprs)
                return flatten(em_pool(em_conv(em_matrix)))

            ########################################################################
            # Constructing entity representation
            em_reprs = []
            for em_inp_embs, em_inp_ebft in zip(em_inp_gembs, em_inp_ebfts):
                em_reprs.append(dropout(mention_repr(em_inp_embs, em_inp_ebft)))
            e_repr = entity_repr(em_reprs)

            # Constructing mention representation
            m_repr = mention_repr(m_inp_embs, m_inp_ebft)

            # Constructing entity-mention representation
            e2m_conv = Convolution2D(nb_filters, 1, nb_filters+embftdim, activation='tanh')
            e2m_pool = MaxPooling2D(pool_size=(2, 1))

            e2m_matrix = Reshape((1, 2, nb_filters+embftdim))(merge([e_repr, m_repr], mode='concat'))
            e2m_repr = flatten(e2m_pool(e2m_conv(e2m_matrix)))

            # Regression
            prob = Dense(1, activation="sigmoid")(e2m_repr)

            # Model compilation
            input_all = sum(em_inp_gembs, []) + em_inp_ebfts + m_inp_embs + [m_inp_ebft]
            self.model = Model(input=input_all, output=prob)
            self.model.compile(loss='mean_squared_error', optimizer='RMSprop')


class EntityMentionGroupFeatCNN_MStackConvRows(AbstractEntityMentionGroupFeatCNN):
    def __init__(self, nb_m4entity, nb_embs, embdim, embftdim, dftdim, nb_filters, gpu_id=-1):
        super(EntityMentionGroupFeatCNN_MStackConvRows, self).__init__()
        self.nb_m4entity, self.nb_embs, self.embdim, self.embftdim, self.dftdim = \
            nb_m4entity, nb_embs, embdim, embftdim, dftdim

        if gpu_id >= 0:
            gpu_options = tf.GPUOptions(visible_device_list=str(gpu_id), allow_growth=True)
            K.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = "-1"

        with tf.device('/cpu:0' if gpu_id < 0 else '/gpu:0'):
            # Constant layers
            flatten, dropout = Flatten(), Dropout(0.65)

            # Input layers
            em_inp_ebfts = [Input(shape=(embftdim,)) for _ in range(nb_m4entity)]
            em_inp_gembs = [[Input(shape=(1, nb_emb, embdim)) for nb_emb in nb_embs] for _ in range(nb_m4entity)]

            m_inp_ebft = Input(shape=(embftdim,))
            m_inp_embs = [Input(shape=(1, nb_emb, embdim)) for nb_emb in nb_embs]

            ########################################################################
            # Mention representation model
            m_rows_reshape = Reshape((1, 3 * len(nb_embs), nb_filters))
            m_ftg_pool = MaxPooling2D(pool_size=(3 * len(nb_embs), 1))
            m_ftg_conv = Convolution2D(nb_filters, 1, nb_filters, activation='tanh')

            m_convs_1r, m_convs_2r, m_convs_3r, m_pools_1r, m_pools_2r, m_pools_3r = [[] for _ in xrange(6)]
            for nb_emb in nb_embs:
                m_convs_1r.append(Convolution2D(nb_filters, 1, embdim, activation='tanh'))
                m_convs_2r.append(Convolution2D(nb_filters, 2, embdim, activation='tanh'))
                m_convs_3r.append(Convolution2D(nb_filters, 3, embdim, activation='tanh'))
                m_pools_1r.append(MaxPooling2D(pool_size=(nb_emb - 0, 1)))
                m_pools_2r.append(MaxPooling2D(pool_size=(nb_emb - 1, 1)))
                m_pools_3r.append(MaxPooling2D(pool_size=(nb_emb - 2, 1)))

            def mention_repr(m_embs, m_ebft):
                gft1rs, gft2rs, gft3rs = [], [], []
                for inp_emb, m_conv_1r, m_conv_2r, m_conv_3r, m_pool_1r, m_pool_2r, m_pool_3r \
                        in zip(m_embs, m_convs_1r, m_convs_2r, m_convs_3r, m_pools_1r, m_pools_2r, m_pools_3r):
                    gft1rs.append(dropout(flatten(m_pool_1r(m_conv_1r(inp_emb)))))
                    gft2rs.append(dropout(flatten(m_pool_2r(m_conv_2r(inp_emb)))))
                    gft3rs.append(dropout(flatten(m_pool_3r(m_conv_3r(inp_emb)))))
                m_ftg_matrix = m_rows_reshape(merge(gft1rs + gft2rs + gft3rs, mode='concat'))
                m_emb_vec = flatten(m_ftg_pool(m_ftg_conv(m_ftg_matrix)))

                return merge([m_emb_vec, m_ebft], mode='concat')

            # Entity representation model
            em_pre_reshape = Reshape((1, nb_m4entity, nb_filters+embftdim))
            em_row_convs = [Convolution2D(nb_filters+embftdim, i+1, nb_filters+embftdim, activation='tanh') for i in xrange(min(3, nb_m4entity))]
            em_row_pools = [MaxPooling2D(pool_size=(nb_m4entity-i, 1)) for i in xrange(min(3, nb_m4entity))]

            em_post_reshape = Reshape((1, len(em_row_convs), nb_filters+embftdim))
            em_all_conv = Convolution2D(nb_filters+embftdim, 1, nb_filters+embftdim, activation='tanh')
            em_all_pool = MaxPooling2D(pool_size=(len(em_row_convs), 1))

            def entity_repr(mention_reprs):
                em_matrix = em_pre_reshape(merge(mention_reprs, mode='concat') if len(mention_reprs) > 1 else mention_reprs)

                em_conv_matrix = []
                for em_conv, em_pool in zip(em_row_convs, em_row_pools):
                    em_conv_matrix.append(dropout(flatten(em_pool(em_conv(em_matrix)))))
                em_conv_matrix = em_post_reshape(merge(em_conv_matrix, mode='concat'))

                return flatten(em_all_pool(em_all_conv(em_conv_matrix)))

            ########################################################################
            # Constructing entity representation
            em_reprs = []
            for em_inp_embs, em_inp_ebft in zip(em_inp_gembs, em_inp_ebfts):
                em_reprs.append(dropout(mention_repr(em_inp_embs, em_inp_ebft)))
            e_repr = entity_repr(em_reprs)

            # Constructing mention representation
            m_repr = mention_repr(m_inp_embs, m_inp_ebft)

            # Constructing entity-mention representation
            e2m_conv = Convolution2D(nb_filters, 1, nb_filters+embftdim, activation='tanh')
            e2m_pool = MaxPooling2D(pool_size=(2, 1))

            e2m_matrix = Reshape((1, 2, nb_filters+embftdim))(merge([e_repr, m_repr], mode='concat'))
            e2m_repr = flatten(e2m_pool(e2m_conv(e2m_matrix)))

            # Regression
            prob = Dense(1, activation="sigmoid")(e2m_repr)

            # Model compilation
            input_all = sum(em_inp_gembs, []) + em_inp_ebfts + m_inp_embs + [m_inp_ebft]
            self.model = Model(input=input_all, output=prob)
            self.model.compile(loss='mean_squared_error', optimizer='RMSprop')


class EntityMentionGroupFeatCNN_MStackPairwise(AbstractEntityMentionGroupFeatCNN):
    def __init__(self, nb_m4entity, nb_embs, embdim, embftdim, dftdim, nb_filters, gpu_id=-1):
        super(EntityMentionGroupFeatCNN_MStackPairwise, self).__init__()
        self.nb_m4entity, self.nb_embs, self.embdim, self.embftdim, self.dftdim = \
            nb_m4entity, nb_embs, embdim, embftdim, dftdim

        if gpu_id >= 0:
            gpu_options = tf.GPUOptions(visible_device_list=str(gpu_id), allow_growth=True)
            K.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = "-1"

        with tf.device('/cpu:0' if gpu_id < 0 else '/gpu:0'):
            # Constant layers
            flatten, dropout = Flatten(), Dropout(0.65)

            # Input layers
            em_inp_ebfts = [Input(shape=(embftdim,)) for _ in range(nb_m4entity)]
            em_inp_gembs = [[Input(shape=(1, nb_emb, embdim)) for nb_emb in nb_embs] for _ in range(nb_m4entity)]

            m_inp_ebft = Input(shape=(embftdim,))
            m_inp_embs = [Input(shape=(1, nb_emb, embdim)) for nb_emb in nb_embs]

            ########################################################################
            # Mention representation model
            m_rows_reshape = Reshape((1, 3 * len(nb_embs), nb_filters))
            m_ftg_pool = MaxPooling2D(pool_size=(3 * len(nb_embs), 1))
            m_ftg_conv = Convolution2D(nb_filters, 1, nb_filters, activation='tanh')

            m_convs_1r, m_convs_2r, m_convs_3r, m_pools_1r, m_pools_2r, m_pools_3r = [[] for _ in xrange(6)]
            for nb_emb in nb_embs:
                m_convs_1r.append(Convolution2D(nb_filters, 1, embdim, activation='tanh'))
                m_convs_2r.append(Convolution2D(nb_filters, 2, embdim, activation='tanh'))
                m_convs_3r.append(Convolution2D(nb_filters, 3, embdim, activation='tanh'))
                m_pools_1r.append(MaxPooling2D(pool_size=(nb_emb - 0, 1)))
                m_pools_2r.append(MaxPooling2D(pool_size=(nb_emb - 1, 1)))
                m_pools_3r.append(MaxPooling2D(pool_size=(nb_emb - 2, 1)))

            def mention_repr(m_embs, m_ebft):
                gft1rs, gft2rs, gft3rs = [], [], []
                for inp_emb, m_conv_1r, m_conv_2r, m_conv_3r, m_pool_1r, m_pool_2r, m_pool_3r \
                        in zip(m_embs, m_convs_1r, m_convs_2r, m_convs_3r, m_pools_1r, m_pools_2r, m_pools_3r):
                    gft1rs.append(dropout(flatten(m_pool_1r(m_conv_1r(inp_emb)))))
                    gft2rs.append(dropout(flatten(m_pool_2r(m_conv_2r(inp_emb)))))
                    gft3rs.append(dropout(flatten(m_pool_3r(m_conv_3r(inp_emb)))))
                m_ftg_matrix = m_rows_reshape(merge(gft1rs + gft2rs + gft3rs, mode='concat'))
                m_emb_vec = flatten(m_ftg_pool(m_ftg_conv(m_ftg_matrix)))

                return merge([m_emb_vec, m_ebft], mode='concat')

            # Mention-mention representation model
            m2m_pre_reshape = Reshape((1, 2, nb_filters+embftdim))
            m2m_conv = Convolution2D(nb_filters+embftdim, 1, nb_filters+embftdim, activation='tanh')
            m2m_pool = MaxPooling2D(pool_size=(2, 1))

            def m2m_repr(m1_vec, m2_vec):
                m2m_matrix = m2m_pre_reshape(merge([m1_vec, m2_vec], mode='concat'))
                return flatten(m2m_pool(m2m_conv(m2m_matrix)))

            # Entity representation model
            nb_mpairs = nb_m4entity * (nb_m4entity - 1) / 2
            em_prow_reshape = Reshape((1, nb_mpairs, nb_filters+embftdim))
            em_prow_conv = Convolution2D(nb_filters+embftdim, 1, nb_filters+embftdim, activation='tanh')
            em_prow_pool = MaxPooling2D(pool_size=(nb_mpairs, 1))

            def entity_repr(mention_reprs):
                m2m_vecs = []
                for idx, m1_repr in enumerate(mention_reprs):
                    for m2_repr in mention_reprs[idx+1:]:
                            m2m_vecs.append(dropout(m2m_repr(m1_repr, m2_repr)))
                mpair_matrix = em_prow_reshape(merge(m2m_vecs, mode='concat'))

                return flatten(em_prow_pool(em_prow_conv(mpair_matrix)))

            ########################################################################
            # Constructing entity representation
            em_reprs = []
            for em_inp_embs, em_inp_ebft in zip(em_inp_gembs, em_inp_ebfts):
                em_reprs.append(dropout(mention_repr(em_inp_embs, em_inp_ebft)))
            e_repr = entity_repr(em_reprs)

            # Constructing mention representation
            m_repr = mention_repr(m_inp_embs, m_inp_ebft)

            # Constructing entity-mention representation
            e2m_conv = Convolution2D(nb_filters, 1, nb_filters+embftdim, activation='tanh')
            e2m_pool = MaxPooling2D(pool_size=(2, 1))

            e2m_matrix = Reshape((1, 2, nb_filters+embftdim))(merge([e_repr, m_repr], mode='concat'))
            e2m_repr = flatten(e2m_pool(e2m_conv(e2m_matrix)))

            # Regression
            prob = Dense(1, activation="sigmoid")(e2m_repr)

            # Model compilation
            input_all = sum(em_inp_gembs, []) + em_inp_ebfts + m_inp_embs + [m_inp_ebft]
            self.model = Model(input=input_all, output=prob)
            self.model.compile(loss='mean_squared_error', optimizer='RMSprop')
