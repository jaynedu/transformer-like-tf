# -*- coding: utf-8 -*-
# @Date    : 2020/9/22 14:26
# @Author  : Du Jing
# @FileName: audio
# ---- Description ----


import os
import sys

import librosa
import numpy as np
import python_speech_features
import tqdm
from pyAudioAnalysis import ShortTermFeatures

import util
from dataset import Dataset

__all_ = ['AudioFeatureExtraction']


class AudioFeatureExtraction(Dataset):
    def __init__(self, path: str):
        try:
            assert os.path.exists(path)
        except:
            sys.stdout.write("[ERROR] 当前数据库路径: %s" % path)
            sys.exit(1)

        super().__init__(path)
        self.frame_length = 400
        self.frame_step = 160

    def preprocess(self, input, use_vad=True):
        output = input
        if use_vad:
            output = os.path.join(r'E:\temp', os.path.basename(input))
            util.Vad().get_audio_with_vad(input, output)

        y, sr = librosa.load(output, sr=16000)

        if use_vad:
            util.clear(output)  # 清除临时文件

        signal = python_speech_features.sigproc.preemphasis(y, 0.97)
        return signal, sr

    def stFeatures(self, signal, sr, frame_length, frame_step):
        features, feature_names = ShortTermFeatures.feature_extraction(signal, sr, frame_length, frame_step)
        features = np.transpose(features)
        return features  # [nFrame (variable), nFeature (fixed)]

    def logfbank(self, signal, sr, frame_length, frame_step):
        winlen = float(frame_length / sr)
        winstep = float(frame_step / sr)
        features = python_speech_features.logfbank(signal, sr, winlen, winstep, nfilt=64)
        return features

    def extract(self, pathList, labelList, feature_func, seq_length=None, use_vad=True):
        paths = []
        labels = []
        features = []
        tbar = tqdm.tqdm(zip(pathList, labelList))
        for path, label in tbar:
            tbar.set_description("FILE: %s" % path)
            try:
                signal, sr = self.preprocess(path, use_vad)
                feature = feature_func(signal, sr, self.frame_length, self.frame_step)
                tbar.set_postfix_str("label: %d, shape: %s" % (label, feature.shape))

                # 如果指定了seq_length，需要对特征进行分割或补零
                if seq_length is not None:
                    length = feature.shape[0]
                    if length <= seq_length:
                        feature = np.pad(feature, ((0, seq_length - length), (0, 0)), 'constant', constant_values=0)
                    else:
                        times = (length - seq_length) // 100 + 1
                        for i in range(times):
                            begin = 100 * i
                            end = begin + seq_length
                            feature = feature[begin: end]

                            features.append(feature)
                            labels.append(label)
                            paths.append(path)

                        # 跳出当前循环
                        continue

                features.append(feature)
                labels.append(label)
                paths.append(path)
            except Exception as error:
                if use_vad:
                    print('[WARNING] - pop: [%s] [%d]' % (path, label))
                print(error)
        print("length:", len(features))
        return paths, labels, features

    def tfrecord_genrator(self, splits, path):
        suffix = ['.train', '.test', '.val']
        for i, split in enumerate(splits):
            x, y = split
            writer = util.createWriter(path + '_' + str(len(x)) + suffix[i])
            for feature, label in zip(x, y):
                util.saveTFrecord(feature, label, writer)
            util.disposeWriter(writer)
