# import numpy as np
# from time import time
# import tensorflow as tf
# from utils.evaluators import BCubeEvaluator
# from keras.layers import Input, Dense, merge
# from sklearn.metrics import mean_squared_error
# from structures.collections import MentionCluster
# from keras.models import Model, save_model, load_model
# from components.features import EntityFeatureExtractor
#
# tf.python.control_flow_ops = tf
#
#
# class FeedForwardModel:
#     def __init__(self, embd_dim, feat_dim, hnode_embd, hnode_feat):
#         e_embd_in = Input(shape=(embd_dim,))
#         m_embd_in = Input(shape=(embd_dim,))
#
#         e_feat_in = Input(shape=(feat_dim,))
#         m_feat_in = Input(shape=(feat_dim,))
#
#         embedding = merge([e_embd_in, m_embd_in], mode='concat', concat_axis=-1)
#         feature = merge([e_feat_in, m_feat_in], mode='concat', concat_axis=-1)
#
#         e_hidden = Dense(hnode_embd, activation='sigmoid')(embedding)
#         f_hidden = Dense(hnode_feat, activation='sigmoid')(feature)
#
#         h_merged = merge([e_hidden, f_hidden], mode='concat', concat_axis=-1)
#         activation = Dense(1, activation='sigmoid')(h_merged)
#
#         self.model = Model(input=[e_embd_in, m_embd_in, e_feat_in, m_feat_in], output=activation)
#         self.model.compile(loss='binary_crossentropy', optimizer='adagrad')
#
#     def fit(self, mentions_trn, mentions_dev, trn_clusters_gold, dev_clusters_gold,
#             Etrn_embd, Mtrn_embd, Etrn_feat, Mtrn_feat, Ytrn_gold,
#             Edev_embd, Mdev_embd, Edev_feat, Mdev_feat, Ydev_gold,
#             nb_epoch=20, batch_size=32, model_out=None):
#
#         best_f1 = best_f1_index = total_time = 0
#
#         trn_p_scores = []
#         trn_r_scores = []
#         trn_f_scores = []
#
#         dev_p_scores = []
#         dev_r_scores = []
#         dev_f_scores = []
#
#         decoder = FeedForwardDecoder()
#         evaluator = BCubeEvaluator()
#
#         for e in range(nb_epoch):
#             start_time = time()
#
#             self.model.fit([Etrn_embd, Mtrn_embd, Etrn_feat, Mtrn_feat], Ytrn_gold,
#                            batch_size=batch_size, nb_epoch=1, shuffle=True, verbose=0)
#
#             Ytrn_pred = self.predict(Etrn_embd, Mtrn_embd, Etrn_feat, Mtrn_feat)
#             trn_error = mean_squared_error(Ytrn_gold, Ytrn_pred)
#
#             Ydev_pred = self.predict(Edev_embd, Mdev_embd, Edev_feat, Mdev_feat)
#             dev_error = mean_squared_error(Ydev_gold, Ydev_pred)
#
#             trn_clusters_pred = decoder.decode(self, mentions_trn)
#             trn_p, trn_r, trn_f1 = evaluator.evaluate(trn_clusters_gold, trn_clusters_pred)
#
#             dev_clusters_pred = decoder.decode(self, mentions_dev)
#             dev_p, dev_r, dev_f1 = evaluator.evaluate(dev_clusters_gold, dev_clusters_pred)
#
#             trn_p_scores.append(trn_p)
#             trn_r_scores.append(trn_r)
#             trn_f_scores.append(trn_f1)
#
#             dev_p_scores.append(dev_p)
#             dev_r_scores.append(dev_r)
#             dev_f_scores.append(dev_f1)
#
#             if best_f1 < dev_f1:
#                 best_f1 = dev_f1
#                 best_f1_index = e
#
#                 if model_out is not None:
#                     self.save_model(model_out)
#
#             lapse = time() - start_time
#             print 'Epoch %3d - Trn loss: %.8f, Dev loss: %.8f, Trn F1: %.8f, Dev F1 : %.8f - %-4.2fs' % \
#                   (e+1, trn_error, dev_error, trn_f1, dev_f1, lapse)
#
#             total_time += lapse
#
#         print '\nTraining Summary:'
#         print 'Best epoch: %d, Trn P/R/F: %.6f/%.6f/%.6f, Dev P/R/F : %.6f/%.6f/%.6f - %-4.2fs' % \
#               (best_f1_index+1,
#                trn_p_scores[best_f1_index], trn_r_scores[best_f1_index], trn_f_scores[best_f1_index],
#                dev_p_scores[best_f1_index], dev_r_scores[best_f1_index], dev_f_scores[best_f1_index],
#                total_time)
#
#         if model_out is not None:
#             print 'Model saved to %s' % model_out
#
#     def predict(self, Etst_embd, Mtst_embd, Etst_feat, Mtst_feat):
#         predictions = self.model.predict([Etst_embd, Mtst_embd, Etst_feat, Mtst_feat])
#         return predictions
#
#     def load_model(self, file_path):
#         try:
#             self.model = load_model(file_path)
#         except IOError:
#             raise IOError("Can't load model file %s" % file_path)
#
#     def save_model(self, file_path):
#         save_model(self.model, file_path)
#
#
# class FeedForwardDecoder(object):
#
#     def decode(self, model, mentions):
#         clusters = [MentionCluster()]
#         e_extractor = EntityFeatureExtractor(len(mentions[0].embedding), len(mentions[0].feature))
#
#         for m in mentions:
#             probs = []
#
#             for c in clusters:
#                 e_embd, e_feat = e_extractor.extract(c)
#
#                 e_embd = e_embd.reshape(1, len(e_embd))
#                 e_feat = e_feat.reshape(1, len(e_feat))
#                 m_embd = m.embedding.reshape(1, len(m.embedding))
#                 m_feat = m.feature.reshape(1, len(m.feature))
#
#                 probs.append(model.predict(e_embd, m_embd, e_feat, m_feat)[0][0])
#
#             target = clusters[np.argmax(probs)]
#
#             if not target:
#                 clusters.append(MentionCluster())
#             target.append(m)
#
#         clusters.pop()
#         return clusters
