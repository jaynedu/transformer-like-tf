# -*- coding: utf-8 -*-
# @Date    : 2020/9/22 14:26
# @Author  : Du Jing
# @FileName: dataset
# ---- Description ----


import contextlib
import random
import json
import os
import tensorflow as tf
import numpy as np
import util
from sklearn.model_selection import train_test_split, StratifiedKFold


class Dataset:
    def __init__(self, path):
        self.path = path

    def load_json(self, delete_labels=None):
        with contextlib.closing(open(self.path, 'r')) as rf:
            content = json.load(rf)['list']
            pathList = []
            labelList = []
            for item in content.values():
                if isinstance(delete_labels, list):
                    if item['label'] not in delete_labels:
                        pathList.append(item['path'])
                        labelList.append(item['label'])
                elif delete_labels is None:
                    pathList.append(item['path'])
                    labelList.append(item['label'])
            return pathList, labelList

    def split_data(self, xs, ys:list):
        count = {}
        for i in set(ys):
            count[i] = ys.count(i)
        maxCount = min(count.values())
        print(maxCount)
        labelCount = {}
        labelCount = labelCount.fromkeys(count.keys(), 0)
        collect = list(zip(xs, ys))
        random.shuffle(collect)
        xs[:], ys[:] = zip(*collect)
        _xs, _ys = [], []
        for i in range(len(xs)):
            if labelCount[ys[i]] < maxCount:
                _xs.append(xs[i])
                _ys.append(ys[i])
                labelCount[ys[i]] += 1
        x_train, x_test, y_train, y_test = train_test_split(_xs, _ys, test_size=0.2, stratify=_ys)
        x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5, stratify=y_test)
        return (x_train, y_train), (x_test, y_test), (x_val, y_val)

    def generate_test_data(self, tfrecord, nfeature, seq_length, total_size):
        test_iterator = util.readTFrecord(tfrecord, nfeature, seq_length, 1, total_size, False)
        x, y, ndim, nframe = test_iterator.get_next()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(test_iterator.initializer)
            inputs, labels, dim, seqlen = sess.run([x, y, ndim, nframe])
            save_path = os.path.splitext(tfrecord)[0] +'.npz'
            np.savez(save_path, inputs=inputs, labels=labels, dim=dim, seqlen=seqlen)

        return save_path
