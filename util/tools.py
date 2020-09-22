# -*- coding: utf-8 -*-
# @Date    : 2020/7/17 8:27 下午
# @Author  : Du Jing
# @FileName: tools
# ---- Description ----
#

import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

__all__ = ['params_usage', 'stats_graph', 'clear', 'check_dir', 'plot_confusion_matrix']


def params_usage(train_variables):
    total = 0
    prompt = []
    for v in train_variables:
        shape = v.get_shape()
        cnt = 1
        for dim in shape:
            cnt *= dim.value
        prompt.append('{} with shape {} has {}'.format(v.name, shape, cnt))
        print(prompt[-1])
        total += cnt
    prompt.append('totaling {}'.format(total))
    print(prompt[-1])
    return '\n'.join(prompt)

def stats_graph(graph):
    flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
    params = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
    print('FLOPs: {};    Trainable params: {}'.format(flops.total_float_ops, params.total_parameters))

def clear(*args):
    for arg in args:
        try:
            os.remove(arg)
        except FileNotFoundError:
            print("文件 [%s] 不存在!" % arg)


def check_dir(dir):
    if not os.path.exists(dir):
        parent = os.path.split(dir)[0]
        check_dir(parent)
        os.mkdir(dir)


def plot_confusion_matrix(matrix, classes, title="Confusion Matrix", save_name="Confusion Matrix.png"):
    plt.figure(figsize=(7, 5.5), dpi=300)
    # 在混淆矩阵中每格的概率值
    matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
    ind_array = np.arange(len(classes))
    x, y = np.meshgrid(ind_array, ind_array)
    thresh = matrix.max() / 2.
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = matrix[y_val][x_val]
        plt.text(x_val, y_val, "%0.2f" % (c,), fontsize=15, va='center', ha='center', color="white" if matrix[x_val, y_val] > thresh else "black")

    plt.imshow(matrix, interpolation='nearest', cmap='Blues')
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, classes, rotation=90)
    plt.yticks(xlocations, classes)
    plt.ylabel('True label')
    plt.xlabel('Predict label')

    # offset the tick
    tick_marks = np.array(range(len(classes))) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)

    # show confusion matrix
    plt.savefig(save_name, format='png')
    plt.show()


def plot_confusion_matrix_direct(matrix_list, classes, title_list, save_path):
    plt.figure(figsize=(14, 14), dpi=300)
    for i, m in enumerate(matrix_list):
        plt.subplot(2, 2, i + 1)
        ind_array = np.arange(len(classes))
        x, y = np.meshgrid(ind_array, ind_array)
        thresh = m.max() / 2.
        for x_val, y_val in zip(x.flatten(), y.flatten()):
            c = m[y_val][x_val]
            plt.text(x_val, y_val, "%0.2f" % (c,), fontsize=15, va='center', ha='center',
                     color="white" if m[x_val, y_val] > thresh else "black")

        plt.imshow(m, interpolation='nearest', cmap='Blues')
        plt.title(title_list[i])
        plt.colorbar()
        xlocations = np.array(range(len(classes)))
        plt.xticks(xlocations, classes, rotation=90)
        plt.yticks(xlocations, classes)
        plt.ylabel('True label')
        plt.xlabel('Predict label')

        # offset the tick
        tick_marks = np.array(range(len(classes))) + 0.5
        plt.gca().set_xticks(tick_marks, minor=True)
        plt.gca().set_yticks(tick_marks, minor=True)
        plt.gca().xaxis.set_ticks_position('none')
        plt.gca().yaxis.set_ticks_position('none')
        plt.grid(True, which='minor', linestyle='-')
        plt.gcf().subplots_adjust(bottom=0.15)
    plt.savefig(save_path, format='png')
    plt.show()


if __name__ == '__main__':
    classes = ['anger', 'happy', 'neutral', 'sad']
    title_list = [
        'Scaled Dot Product Attention - URDU',
        'Proposed Attention - URDU',
        'Scaled Dot Product Attention - CASIA',
        'Proposed Attention - CASIA',
    ]
    save_path = r'../results/summarize_confusion_matrix.png'
    m0 = np.array(
        [[0.9, 0.0, 0.1, 0.0],
         [0.1, 0.7, 0.0, 0.2],
         [0.0, 0.1, 0.7, 0.2],
         [0.0, 0.1, 0.0, 0.9]]
    )
    m1 = np.array(
        [[0.9, 0.1, 0.0, 0.0],
         [0.0, 0.6, 0.2, 0.2],
         [0.1, 0.1, 0.7, 0.1],
         [0.0, 0.1, 0.2, 0.7]]
    )
    m2 = np.array(
        [[0.93, 0.03, 0.00, 0.04],
         [0.03, 0.92, 0.02, 0.03],
         [0.02, 0.05, 0.93, 0.00],
         [0.01, 0.04, 0.01, 0.94]]
    )
    m3 = np.array(
        [[0.87, 0.10, 0.02, 0.01],
         [0.13, 0.79, 0.06, 0.02],
         [0.07, 0.03, 0.89, 0.01],
         [0.01, 0.04, 0.01, 0.94]]
    )
    m_list = [m0, m1, m2, m3]
    plot_confusion_matrix_direct(m_list, classes, title_list, save_path)
