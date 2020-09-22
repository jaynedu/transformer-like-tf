# -*- coding: utf-8 -*-
# @Date: 2020/7/22 12:41
# @Author: Du Jing
# @FileName: train
# ---- Description ----
#

import os
import time

import numpy as np
import tensorflow as tf

# 环境设置
np.set_printoptions(threshold=np.inf)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.device('/gpu:0')
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # 按需加内存
datetime = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))

# 模型版本控制
model_name = os.path.split(os.path.dirname(__file__))[-1]  # 当前模型文件夹名称
position_encoding_type = "concat"
attention_unit_type = "tayler"
feature_name = "logfbank"
dataset_name = "des"

model_version = '_'.join([model_name, position_encoding_type, attention_unit_type, feature_name, dataset_name])
model_save_dir = os.path.join(r'E:\Models', model_name, model_version)

# 路径设置
train_tensorboard_path = os.path.join('logs', datetime + '_' + model_version, 'train')
val_tensorboard_path = os.path.join('logs', datetime + '_' + model_version, 'val')
train_path = r'E:\Datasets\des_512_246.train'
val_path = r'E:\Datasets\des_512_31.val'
test_path = r'E:\Datasets\des_512_31.test'

# 训练参数设置
train_size = eval(os.path.splitext(train_path)[0].split('_')[-1])
val_size = eval(os.path.splitext(val_path)[0].split('_')[-1])
test_size = eval(os.path.splitext(test_path)[0].split('_')[-1])
train_batch = 64
val_batch = val_size
epoch = 500
eta = 0.001
warmup = 1000.

# 模型参数设置
nlabel = 4
nfeature = 64
seq_length = 512
position_size = 64
head_num = 8
head_size = int(128 / head_num) if position_encoding_type == 'concat' else int(64 / head_num)
feed_forward_size = [512, 128] if position_encoding_type == 'concat' else [256, 64]
dropout = 0.1

# 测试参数控制
classes = ['anger', 'neutral', 'happy', 'sad']
# classes = ["agressiv", "neutral", "cheerful", "tired"]  # abc
plot_matrix = True
figure_title = '_'.join([position_encoding_type, attention_unit_type, dataset_name, str(head_size)])
figure_save_path = r"fig_" + model_version + ".png"
