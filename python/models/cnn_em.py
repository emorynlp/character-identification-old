import numpy as np
from time import time
import tensorflow as tf
from utils.evaluators import BCubeEvaluator
from sklearn.metrics import mean_squared_error
from structures.collections import MentionCluster
from components.features import EntityFeatureExtractor
from keras.models import Model, save_model, load_model
from keras.layers import Input, Reshape, Flatten, Dense, merge
from keras.layers.convolutional import Convolution2D, MaxPooling2D

tf.python.control_flow_ops = tf


class EntityMentionCNN:
    def __init__(self, entity_chns, nb_emb_feats, embdim, dftdim, conv_rows, emb_nb_filters):
        self.entity_chns = entity_chns
        self.nb_emb_feats = nb_emb_feats
        self.embdim, self.dftdim = embdim, dftdim

        # Input: [e_embedding*chns] + [m_embedding, e_features*chns(chns*dftdim), m_features]
        inp_layers = []

        # Entity embedding vector
        em_vectors = []
        for i in xrange(entity_chns):
            inp_em = Input(shape=(1, nb_emb_feats, embdim))
            conv_em = Convolution2D(emb_nb_filters, conv_rows, embdim, activation='tanh')(inp_em)
            pool_em = MaxPooling2D(pool_size=(nb_emb_feats - conv_rows + 1, 1))(conv_em)
            em_vector = Reshape((emb_nb_filters, 1))(pool_em)

            inp_layers.append(inp_em)
            em_vectors.append(em_vector)

        e_matrix = Reshape((1, entity_chns, emb_nb_filters))(merge(em_vectors, mode='concat'))
        conv_e = Convolution2D(emb_nb_filters, conv_rows, emb_nb_filters, activation='tanh')(e_matrix)
        pool_e = MaxPooling2D(pool_size=(entity_chns - conv_rows + 1, 1))(conv_e)
        e_vector = Reshape((emb_nb_filters,))(Flatten()(pool_e))

        # Mention embedding vector
        inp_m = Input(shape=(1, nb_emb_feats, embdim))
        inp_layers.append(inp_m)

        conv_m = Convolution2D(emb_nb_filters, conv_rows, embdim, activation='tanh')(inp_m)
        pool_m = MaxPooling2D(pool_size=(nb_emb_feats - conv_rows + 1, 1))(conv_m)
        m_vector = Reshape((emb_nb_filters,))(pool_m)

        # Entity feature vector
        inp_ft_e = Input(shape=(1, entity_chns, dftdim))
        inp_layers.append(inp_ft_e)

        conv_ft_e = Convolution2D(dftdim, conv_rows, dftdim, activation='tanh')(inp_ft_e)
        pool_ft_e = MaxPooling2D(pool_size=(entity_chns - conv_rows + 1, 1))(conv_ft_e)
        e_ft_vector = Reshape((dftdim,))(pool_ft_e)

        # Mention feature vector
        m_ft_vector = Input(shape=(dftdim,))
        inp_layers.append(m_ft_vector)

        # Entity-Mention embedding & feature vectors
        emb_vector = merge([e_vector, m_vector], mode='concat')
        ft_vector = merge([e_ft_vector, m_ft_vector], mode='concat')

        # Regression
        reg_emb = Dense(1, activation="sigmoid")(emb_vector)
        reg_ft = Dense(1, activation="sigmoid")(ft_vector)
        prob = Dense(1, activation="sigmoid")(merge([reg_emb, reg_ft], mode="concat"))

        # Model compilation
        self.model = Model(input=inp_layers, output=prob)
        self.model.compile(loss='mse', optimizer='adagrad')

    def fit(self, mentions_trn, mentions_dev, trn_clusters_gold, dev_clusters_gold,
            Xtrn, Ytrn, Xdev, Ydev, nb_epoch=20, batch_size=32, model_out=None):

        best_trn_scores, best_dev_scores, best_epoch, total_time = ([0]*3, [0]*3, 0, 0)
        decoder = EntityMentionCNNDecoder()
        evaluator = BCubeEvaluator()

        for e in range(nb_epoch):
            start_time = time()
            self.model.fit(Xtrn, Ytrn, batch_size=batch_size, nb_epoch=1, shuffle=True, verbose=0)
            print "Model fitting          - %.2fs" % (time() - start_time)

            start_time = time()
            trn_error = mean_squared_error(Ytrn, self.predict(Xtrn))
            dev_error = mean_squared_error(Ydev, self.predict(Xdev))
            print "Loss calculations      - %.2fs" % (time() - start_time)

            # trn_clusters_pred = decoder.decode(self, mentions_trn)
            trn_p, trn_r, trn_f1 = 0, 0, 0
            # trn_p, trn_r, trn_f1 = evaluator.evaluate(trn_clusters_gold, trn_clusters_pred)

            start_time = time()
            dev_clusters_pred = decoder.decode(self, mentions_dev)
            print "Dev cluster decoding   - %.2fs" % (time() - start_time)
            # dev_p, dev_r, dev_f1 = 0, 0, 0
            start_time = time()
            dev_p, dev_r, dev_f1 = evaluator.evaluate(dev_clusters_gold, dev_clusters_pred)
            print "Dev cluster evaluation - %.2fs" % (time() - start_time)
            print "Dev: Found %d clusters" % len(dev_clusters_pred)

            if best_dev_scores[2] < dev_f1:
                best_epoch = e
                best_dev_scores = [dev_p, dev_r, dev_f1]
                best_trn_scores = [trn_p, trn_r, trn_f1]

                if model_out is not None:
                    self.save_model(model_out)

            lapse = time() - start_time
            total_time += lapse

            print 'Epoch %3d - Trn loss: %.8f, Dev loss: %.8f, Trn F1: %.8f, Dev F1 : %.8f - %4.2fs' % \
                  (e + 1, trn_error, dev_error, trn_f1, dev_f1, lapse)

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


class EntityMentionCNNDecoder(object):

    def decode(self, model, mentions):
        extractor = EntityFeatureExtractor(mentions[0].embedding.shape, mentions[0].feature.shape)
        clusters = [MentionCluster()]

        extraction_time, prediction_time = 0, 0
        for mention in mentions:
            m_embedding = mention.embedding.reshape(1, model.nb_emb_feats, model.embdim)
            m_feature   = mention.feature

            e_embeddings, m_embeddings, e_features, m_features, probs = ([], [], [], [], [])

            start_time = time()
            for cluster in clusters:
                e_embedding, e_feature = extractor.extract(cluster, True, model.entity_chns)
                e_embeddings.append(e_embedding)
                e_features.append(e_feature.reshape(1, model.entity_chns, model.dftdim))

                m_embeddings.append(m_embedding)
                m_features.append(m_feature)

            e_embeddings = np.swapaxes(e_embeddings, 0, 1)
            e_embeddings = [es.reshape(len(clusters), 1, model.nb_emb_feats, model.embdim) for es in e_embeddings]

            instances = e_embeddings + [np.array(m_embeddings), np.array(e_features), np.array(m_features)]
            extraction_time += time() - start_time

            start_time = time()
            probs = model.predict(instances)
            target = clusters[np.argmax(probs.reshape(1, len(clusters)))]
            prediction_time += time() - start_time

            if not target:
                clusters.append(MentionCluster())
            target.append(mention)

        print "\tCluster feature extraction - %.2fs" % extraction_time
        print "\tCluster-Mention prediction - %.2fs" % prediction_time

        clusters.pop()
        return clusters
