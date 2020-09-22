# -*- coding: utf-8 -*-
# @Date    : 2020/9/22 14:27
# @Author  : Du Jing
# @FileName: preprocess
# ---- Description ----

import audio

# parameters
json_file = 'dataset/des.json'  # 原始数据库json文件
delete_labels = [4, 5, 6, 7, 8, 9]  # 不计入数据集的标签
save_path = r'E:\Datasets\des_512'

extractor = audio.AudioFeatureExtraction(json_file)

print("加载JSON...")
paths, labels = extractor.load_json(delete_labels)

print("正在提取特征...")
# paths, labels, features = extractor.extract(paths, labels, extractor.stFeatures, seq_length=300, use_vad=False)
paths, labels, features = extractor.extract(paths, labels, extractor.logfbank, seq_length=512, use_vad=False)

print("正在划分数据集...")
splits = extractor.split_data(features, labels)

print("正在生成数据集...")
extractor.tfrecord_genrator(splits, save_path)