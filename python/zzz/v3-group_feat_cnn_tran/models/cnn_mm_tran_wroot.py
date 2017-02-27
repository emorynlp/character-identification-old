import os
import numpy as np
from time import time
import tensorflow as tf
from keras import backend as K
from sklearn.metrics import accuracy_score
from utils.evaluators import BCubeEvaluator
from keras.utils.np_utils import to_categorical
from structures.collections import MentionCluster
from keras.models import Model, save_model, load_model
from components.features import MentionPairFeatureExtractor
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers import Input, Reshape, Flatten, Dense, merge, Dropout

tf.logging.set_verbosity(tf.logging.WARN)
tf.python.control_flow_ops = tf


# LRS: Left-Reduced-Shift, P: Pass, S: Shift (Head mention), R: Reduce (Singleton)
class LABELS(object):
    LRS, P, R, S, nb_labels = range(5)
    _m = {0: 'LRS',  1: 'P', 2: 'R', 3: 'S', 'LRS': 0, 'P': 1, 'R': 2, 'S': 3}

    @staticmethod
    def to_class_id(label):
        return LABELS._m.get(label.upper(), None)

    @staticmethod
    def to_class_label(class_id):
        return LABELS._m.get(int(class_id), None)


class MentionMentionCNN:
    def __init__(self, nb_embs, embdim, embftdim, dftdim, nb_filters, gpu_id=-1):
        self.nb_embs, self.embdim, self.embftdim, self.dftdim = nb_embs, embdim, embftdim, dftdim

        if gpu_id >= 0:
            gpu_options = tf.GPUOptions(visible_device_list=str(gpu_id), allow_growth=True)
            K.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = "-1"

        with tf.device('/cpu:0' if gpu_id < 0 else '/gpu:0'):
            # Input layers
            mm_inp_dft = Input(shape=(dftdim,))
            m1_inp_embs = [Input(shape=(1, nb_emb, embdim)) for nb_emb in nb_embs]
            m2_inp_embs = [Input(shape=(1, nb_emb, embdim)) for nb_emb in nb_embs]
            m1_inp_ebft, m2_inp_ebft = Input(shape=(embftdim,)), Input(shape=(embftdim,))

            # Convolution + Pooling layers
            convs_1r, convs_2r, convs_3r, pools_1r, pools_2r, pools_3r = [], [], [], [], [], []
            for nb_emb in nb_embs:
                convs_1r.append(Convolution2D(nb_filters, 1, embdim, activation='tanh'))
                convs_2r.append(Convolution2D(nb_filters, 2, embdim, activation='tanh'))
                convs_3r.append(Convolution2D(nb_filters, 3, embdim, activation='tanh'))
                pools_1r.append(MaxPooling2D(pool_size=(nb_emb - 0, 1)))
                pools_2r.append(MaxPooling2D(pool_size=(nb_emb - 1, 1)))
                pools_3r.append(MaxPooling2D(pool_size=(nb_emb - 2, 1)))

            # Mention-mention embedding vectors
            reshape, dropout = Reshape((nb_filters,)), Dropout(0.8)
            m1_emb_vec_1r, m1_emb_vec_2r, m1_emb_vec_3r, m2_emb_vec_1r, m2_emb_vec_2r, m2_emb_vec_3r = [], [], [], [], [], []
            for m1_inp, m2_inp, conv_1r, conv_2r, conv_3r, pool_1r, pool_2r, pool_3r in \
                    zip(m1_inp_embs, m2_inp_embs, convs_1r, convs_2r, convs_3r, pools_1r, pools_2r, pools_3r):
                m1_emb_vec_1r.append(dropout(reshape(pool_1r(conv_1r(m1_inp)))))
                m1_emb_vec_2r.append(dropout(reshape(pool_2r(conv_2r(m1_inp)))))
                m1_emb_vec_3r.append(dropout(reshape(pool_3r(conv_3r(m1_inp)))))
                m2_emb_vec_1r.append(dropout(reshape(pool_1r(conv_1r(m2_inp)))))
                m2_emb_vec_2r.append(dropout(reshape(pool_2r(conv_2r(m2_inp)))))
                m2_emb_vec_3r.append(dropout(reshape(pool_3r(conv_3r(m2_inp)))))

            nb_rows = sum([len(m1_emb_vec_1r), len(m1_emb_vec_2r), len(m1_emb_vec_3r)])
            m1_matrix_emb = Reshape((1, nb_rows, nb_filters))(merge(m1_emb_vec_1r + m1_emb_vec_2r + m1_emb_vec_3r, mode='concat'))
            m2_matrix_emb = Reshape((1, nb_rows, nb_filters))(merge(m2_emb_vec_1r + m2_emb_vec_2r + m2_emb_vec_3r, mode='concat'))
            conv_m = Convolution2D(nb_filters, 1, nb_filters, activation='tanh')
            pool_m = MaxPooling2D(pool_size=(nb_rows, 1))

            m1_vec = merge([Reshape((nb_filters,))(pool_m(conv_m(m1_matrix_emb))), m1_inp_ebft], mode='concat')
            m2_vec = merge([Reshape((nb_filters,))(pool_m(conv_m(m2_matrix_emb))), m2_inp_ebft], mode='concat')

            mm_matrix_vec = Reshape((1, 2, nb_filters+embftdim))(merge([m1_vec, m2_vec], mode='concat'))
            conv_mm = Convolution2D(nb_filters, 1, nb_filters, activation='tanh')
            pool_mm = MaxPooling2D(pool_size=(2, 1))
            mm_vec = merge([Flatten()(pool_mm(conv_mm(mm_matrix_vec))), mm_inp_dft], mode='concat')

            # Regression
            probs = Dense(LABELS.nb_labels, activation="sigmoid")(mm_vec)

        # Model compilation
        self.model = Model(input=m1_inp_embs + m2_inp_embs + [m1_inp_ebft, m2_inp_ebft, mm_inp_dft], output=probs)
        self.model.compile(loss='categorical_crossentropy', optimizer='RMSprop')

    def fit(self, mentions_trn, mentions_dev, trn_cluster_docs_gold, dev_cluster_docs_gold,
            Xtrn, Ytrn, Xdev, Ydev, eval_every=1, nb_epoch=20, batch_size=32, model_out=None):

        best_trn_scores, best_dev_scores, best_epoch, total_time = ([0]*3, [0]*3, 0, 0)
        decoder = MentionMentionCNNDecoder()
        evaluator = BCubeEvaluator()

        for e in range(nb_epoch/eval_every):
            global_start_time = local_start_time = time()
            self.model.fit(Xtrn, to_categorical(Ytrn, LABELS.nb_labels),
                           batch_size=batch_size, nb_epoch=eval_every, shuffle=True, verbose=0)
            trn_time = time() - local_start_time

            trn_accuracy = accuracy_score(Ytrn, self.predict(Xtrn))
            dev_accuracy = accuracy_score(Ydev, self.predict(Xdev))

            local_start_time = time()
            trn_cluster_docs_pred = decoder.decode(self, mentions_trn)
            trn_p, trn_r, trn_f1 = evaluator.evaluate_docs(trn_cluster_docs_gold, trn_cluster_docs_pred)

            dev_cluster_docs_pred = decoder.decode(self, mentions_dev)
            dev_p, dev_r, dev_f1 = evaluator.evaluate_docs(dev_cluster_docs_gold, dev_cluster_docs_pred)
            decode_time = time() - local_start_time

            if best_dev_scores[2] < dev_f1:
                best_epoch = e
                best_dev_scores = [dev_p, dev_r, dev_f1]
                best_trn_scores = [trn_p, trn_r, trn_f1]

                if model_out is not None:
                    self.save_model(model_out)

            lapse = time() - global_start_time
            total_time += lapse

            print 'Epoch %3d - Trn Accu(P/R/F): %.4f(%.4f/%.4f/%.4f), Dev Accu(P/R/F): %.4f(%.4f/%.4f/%.4f) - %4.2fs'\
                  % ((e+1)*eval_every, trn_accuracy, trn_p, trn_r, trn_f1, dev_accuracy, dev_p, dev_r, dev_f1, lapse)
            print '\tTime breakdown: Train %.2f, Decode %.2f' % (trn_time, decode_time)

        '\nTraining Summary:'
        print 'Best epoch: %d, Trn P/R/F: %.6f/%.6f/%.6f, Dev P/R/F : %.6f/%.6f/%.6f - %4.2fs' % \
              ((best_epoch+1)*eval_every, best_trn_scores[0], best_trn_scores[1], best_trn_scores[2],
               best_dev_scores[0], best_dev_scores[1], best_dev_scores[2], total_time)

        if model_out is not None:
            print 'Model saved to %s' % model_out

    def decode(self, mention_docs):
        return MentionMentionCNNDecoder().decode(self, mention_docs)

    def predict(self, Xtst):
        return np.argmax(self.model.predict(Xtst), axis=-1)

    def load_model(self, file_path):
        try:
            self.model = load_model(file_path)
        except IOError:
            raise IOError("Can't load model file %s" % file_path)

    def save_model(self, file_path):
        save_model(self.model, file_path)


class MentionMentionCNNDecoder(object):
    def __init__(self):
        self.mm_extractor = MentionPairFeatureExtractor()

    def decode(self, model, mention_docs):
        cluster_docs, label_counts = [], dict()
        for mentions in mention_docs:
            m_m2cluster, m_m2fs, valid_prev = dict(), dict(), [mentions[0]]

            for m in mentions:
                emb = [eb.reshape(1, 1, neb, model.embdim) for eb, neb in zip(m.embedding, model.nb_embs)]
                ebft = m.feature.reshape(1, model.embftdim)
                m_m2fs[m] = (emb, ebft)

            for curr in mentions[1:]:
                curr_emb, curr_ebft = m_m2fs[curr]
                target, v_count = None, 0

                for prev in reversed(valid_prev):
                    v_count += 1

                    prev_emb, prev_ebft = m_m2fs[prev]
                    mm_dft = self.mm_extractor.extract((prev, curr)).reshape(1, model.dftdim)
                    instance = prev_emb + curr_emb + [prev_ebft, curr_ebft, mm_dft]

                    label = np.asscalar(model.predict(instance)[0])
                    label_counts[label] = label_counts.get(label, 0) + 1
                    if label is not LABELS.P:

                        if label == LABELS.LRS:
                            target = m_m2cluster[prev]
                            valid_prev.insert(len(valid_prev)-1, curr)
                        elif label == LABELS.S:
                            valid_prev.insert(len(valid_prev)-1, curr)
                        break

                    # If labels for previous mentions are all "PASS"
                    if v_count == len(valid_prev):
                        valid_prev.insert(len(valid_prev)-1, curr)

                if target is None:
                    target = MentionCluster()
                target.append(curr)

                m_m2cluster[curr] = target
            cluster_docs.append(list(set(m_m2cluster.values())))

        print dict([(LABELS.to_class_label(e[0]), e[1]) for e in label_counts.items()])
        return cluster_docs
