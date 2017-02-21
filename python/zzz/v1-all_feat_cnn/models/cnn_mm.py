from time import time
import tensorflow as tf
from keras.optimizers import Adagrad
from utils.evaluators import BCubeEvaluator
from sklearn.metrics import mean_squared_error
from structures.collections import MentionCluster
from keras.models import Model, save_model, load_model
from keras.layers import Input, Reshape, Flatten, Dense, merge
from keras.layers.convolutional import Convolution2D, MaxPooling2D

tf.python.control_flow_ops = tf


class MentionMentionCNN:
    def __init__(self, nb_emb_feats, embdim, dftdim, emb_nb_filters, dft_nb_filters):
        self.nb_emb_feats = nb_emb_feats
        self.emb_nb_filters = emb_nb_filters
        self.dft_nb_filters = dft_nb_filters
        self.embdim, self.dftdim = embdim, dftdim

        # Mention-Mention embedding vector
        inp_m1_emb = Input(shape=(1, nb_emb_feats, embdim))
        inp_m2_emb = Input(shape=(1, nb_emb_feats, embdim))

        conv_m_emb_1r = Convolution2D(emb_nb_filters, 1, embdim, activation='tanh')
        pool_m_emb_1r = MaxPooling2D(pool_size=(nb_emb_feats, 1))
        emb_m1_vector_1r = Reshape((emb_nb_filters,))(pool_m_emb_1r(conv_m_emb_1r(inp_m1_emb)))
        emb_m2_vector_1r = Reshape((emb_nb_filters,))(pool_m_emb_1r(conv_m_emb_1r(inp_m2_emb)))

        conv_m_emb_2r = Convolution2D(emb_nb_filters, 2, embdim, activation='tanh')
        pool_m_emb_2r = MaxPooling2D(pool_size=(nb_emb_feats-1, 1))
        emb_m1_vector_2r = Reshape((emb_nb_filters,))(pool_m_emb_2r(conv_m_emb_2r(inp_m1_emb)))
        emb_m2_vector_2r = Reshape((emb_nb_filters,))(pool_m_emb_2r(conv_m_emb_2r(inp_m2_emb)))

        conv_m_emb_3r = Convolution2D(emb_nb_filters, 3, embdim, activation='tanh')
        pool_m_emb_3r = MaxPooling2D(pool_size=(nb_emb_feats-2, 1))
        emb_m1_vector_3r = Reshape((emb_nb_filters,))(pool_m_emb_3r(conv_m_emb_3r(inp_m1_emb)))
        emb_m2_vector_3r = Reshape((emb_nb_filters,))(pool_m_emb_3r(conv_m_emb_3r(inp_m2_emb)))

        merged_vectors = merge([emb_m1_vector_1r, emb_m2_vector_1r, emb_m1_vector_2r, emb_m2_vector_2r, emb_m1_vector_3r, emb_m2_vector_3r], mode='concat')
        emb_m_matrix = Reshape((1, 2, emb_nb_filters))(merged_vectors)
        conv_mm_emb = Convolution2D(emb_nb_filters, 1, emb_nb_filters, activation='tanh')(emb_m_matrix)
        emb_mm_vector = Reshape((emb_nb_filters,))(Flatten()(MaxPooling2D(pool_size=(2, 1))(conv_mm_emb)))

        # Mention-Mention feature vector
        inp_m1_dft = Input(shape=(dftdim,))
        inp_m2_dft = Input(shape=(dftdim,))

        inp_mm_dft = Reshape((1, 2, dftdim))(merge([inp_m1_dft, inp_m2_dft], mode='concat'))
        conv_mm_dft = Convolution2D(dft_nb_filters, 1, dftdim, activation='tanh')(inp_mm_dft)
        dft_mm_vector = Reshape((dft_nb_filters,))(Flatten()(MaxPooling2D(pool_size=(2, 1))(conv_mm_dft)))

        # Regression
        prob = Dense(1, activation="sigmoid")(merge([emb_mm_vector, dft_mm_vector], mode="concat"))

        # Model compilation
        self.model = Model(input=[inp_m1_emb, inp_m2_emb, inp_m1_dft, inp_m2_dft], output=prob)
        self.model.compile(loss='mse', optimizer=Adagrad(lr=0.08))

    def fit(self, mentions_trn, mentions_dev, trn_cluster_docs_gold, dev_cluster_docs_gold,
            Xtrn, Ytrn, Xdev, Ydev, eval_every=1, nb_epoch=20, batch_size=32, model_out=None):

        best_trn_scores, best_dev_scores, best_epoch, total_time = ([0]*3, [0]*3, 0, 0)
        decoder = MentionMentionCNNDecoder()
        evaluator = BCubeEvaluator()

        for e in range(nb_epoch):
            global_start_time = time()
            self.model.fit(Xtrn, Ytrn, batch_size=batch_size, nb_epoch=1, shuffle=True, verbose=0)

            trn_error = mean_squared_error(Ytrn, self.predict(Xtrn))
            dev_error = mean_squared_error(Ydev, self.predict(Xdev))
            trn_p, trn_r, trn_f1, dev_p, dev_r, dev_f1 = 0, 0, 0, 0, 0, 0

            eval = (e + 1) % eval_every == 0
            if eval:
                trn_cluster_docs_pred = decoder.decode(self, mentions_trn)
                trn_p, trn_r, trn_f1 = evaluator.evaluate_docs(trn_cluster_docs_gold, trn_cluster_docs_pred)

                dev_cluster_docs_pred = decoder.decode(self, mentions_dev)
                dev_p, dev_r, dev_f1 = evaluator.evaluate_docs(dev_cluster_docs_gold, dev_cluster_docs_pred)

                if best_dev_scores[2] < dev_f1:
                    best_epoch = e
                    best_dev_scores = [dev_p, dev_r, dev_f1]
                    best_trn_scores = [trn_p, trn_r, trn_f1]

                    if model_out is not None:
                        self.save_model(model_out)

            lapse = time() - global_start_time
            total_time += lapse

            if eval:
                print '>> Epoch %3d - Trn P/R/F: %.4f/%.4f/%.4f, Dev P/R/F : %.4f/%.4f/%.4f - %4.2fs\n' % \
                      (e + 1, trn_p, trn_r, trn_f1, dev_p, dev_r, dev_f1, lapse)
            else:
                print '   Epoch %3d - Trn loss: %.8f, Dev loss: %.8f - %4.2fs' % (e + 1, trn_error, dev_error, lapse)

        '\nTraining Summary:'
        print 'Best epoch: %d, Trn P/R/F: %.6f/%.6f/%.6f, Dev P/R/F : %.6f/%.6f/%.6f - %4.2fs' % \
              (best_epoch + 1, best_trn_scores[0], best_trn_scores[1], best_trn_scores[2],
               best_dev_scores[0], best_dev_scores[1], best_dev_scores[2], total_time)

        if model_out is not None:
            print 'Model saved to %s' % model_out

    def predict(self, Xtst):
        return self.model.predict(Xtst)

    def load_model(self, file_path):
        try:
            self.model = load_model(file_path)
        except IOError:
            raise IOError("Can't load model file %s" % file_path)

    def save_model(self, file_path):
        save_model(self.model, file_path)


class MentionMentionCNNDecoder(object):
    def decode(self, model, mention_docs):
        cluster_docs = []
        for mentions in mention_docs:
            m_m2cluster = dict()
            m_m2cluster[mentions[0]] = MentionCluster([mentions[0]])

            for c_idx, curr in enumerate(mentions[1:], 1):
                curr_emb = curr.embedding.reshape(1, 1, model.nb_emb_feats, model.embdim)
                curr_dft = curr.feature.reshape(1, model.dftdim)

                target = None
                for prev in reversed(mentions[:c_idx]):
                    prev_emb = prev.embedding.reshape(1, 1, model.nb_emb_feats, model.embdim)
                    prev_dft = prev.feature.reshape(1, model.dftdim)

                    prob = model.predict([prev_emb, curr_emb, prev_dft, curr_dft])[0][0]
                    if prob > 0.5:
                        target = m_m2cluster[prev]
                        break

                if target is None:
                    target = MentionCluster()
                target.append(curr)
                m_m2cluster[curr] = target
            cluster_docs.append(list(set(m_m2cluster.values())))

        return cluster_docs
