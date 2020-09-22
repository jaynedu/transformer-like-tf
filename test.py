# -*- coding: utf-8 -*-
# @Date: 2020/8/5 13:58
# @Author: Du Jing
# @FileName: plot
# ---- Description ----
#

import sys

import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

import util
from dataset import Dataset
import args
from graph import model

# 加载数据集
test_data_path = Dataset(None).generate_test_data(args.test_path, args.nfeature, args.seq_length, args.test_size)
test_data = np.load(test_data_path)
testData, testLabel, testx, testy = test_data['inputs'], test_data['labels'], test_data['seqlen'], test_data['dim'],
feed_dict_test = {model.x_input: testData,
                  model.y_true: testLabel,
                  model.seqLen: testx,
                  model.dropout: 0,
                  model.training: False}

saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    if tf.train.checkpoint_exists(args.model_save_dir):
        model_file = tf.train.latest_checkpoint(args.model_save_dir)
        saver.restore(sess, model_file)
        y_pred_test, test_acc, test_loss = sess.run([model.y_pred, model.accuracy, model.loss],
                                                    feed_dict=feed_dict_test)
        mhas = sess.run([model.mhas], feed_dict_test)[0]
        plt.figure(dpi=300, figsize=(7.6, 12.8))
        for i, att in enumerate(mhas):
            print(mhas[i][0].shape)
            plt.subplot(6, 1, i+1)
            plt.imshow(np.transpose(att[0]))
        plt.show()

        print(classification_report(y_true=testLabel, y_pred=y_pred_test))
        matrix = confusion_matrix(y_true=testLabel, y_pred=y_pred_test)
        print(matrix)
        if args.plot_matrix:
            util.plot_confusion_matrix(matrix, args.classes, args.figure_title, args.figure_save_path)
    else:
        sys.stderr.write('Checkpoint Not Found!')
        sys.exit(1)
