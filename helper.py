import numpy as np
import tensorflow as tf
import sklearn.metrics
import json
import argparse
import random
import time
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import gen_random_ops
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed

def norm_array(array):
    min = np.reshape(np.min(array, 1), [-1,1])
    max = np.reshape(np.max(array, 1), [-1,1])
    return (array-min)/(max-min)

def linear(input, output_size, input_size = None, name=None, reuse=False, if_bias=True, stddev=0.1):
    with tf.variable_scope(name or "Linear") as scope:
        if reuse:
            scope.reuse_variables()
        if input_size==None:
            input_size = input.get_shape().as_list()
        matrix = tf.get_variable("kernel", [input_size[-1], output_size], tf.float32, tf.random_normal_initializer(stddev=stddev))
        if if_bias:
            bias = tf.get_variable("bias", [output_size], initializer=tf.constant_initializer(0.0))
            return tf.matmul(input, matrix) + bias
        else:
            return tf.matmul(input, matrix)

def data_loader(dataset):
    data_path = '/data/dongxue/recom/%s/'%dataset
    item_data = json.load(open(data_path + 'item_data.json', 'r'))
    train_pair = json.load(open(data_path + 'train_pair.json', 'r'))
    valid_pair = json.load(open(data_path + 'valid_pair.json', 'r'))
    test_pair = json.load(open(data_path + 'test_pair.json', 'r'))
    user_idx = json.load(open(data_path + 'user_idx.json', 'r'))
    item_idx = json.load(open(data_path + 'item_idx.json', 'r'))
    voc = json.load(open(data_path + 'voc.json', 'r'))

    user_dict, item_dict = {}, {}
    for idx,i in enumerate(user_idx):
        user_dict[i] = idx
    for idx,i in enumerate(item_idx):
        item_dict[i] = idx

    text_dict = {}
    for item_i in list(item_data.keys()):
        textEmb = np.zeros([len(voc)])
        text = item_data[item_i]['words']
        for word in text:
            if word in voc:
                textEmb[voc.index(word)] = 1.
        text_dict[item_i] = textEmb
    return data_path,item_data,train_pair,valid_pair,test_pair,user_dict,item_dict,voc,text_dict


def pocs_batch(model, batch, user_idx, item_idx, text_dict, step):
    u, i, k, ti, tk = [], [], [], [], []
    for _, outfit_i in enumerate(batch):
        user_i, item_i = outfit_i
        random_item = random.sample(batch, 1)[0][-1]
        u.append(user_idx[user_i])
        i.append(item_idx[item_i])
        k.append(item_idx[random_item])
        ti.append(text_dict[item_i])
        tk.append(text_dict[random_item])
    feed_dict = {model.u: u, model.i: i, model.k: k, model.ti: ti, model.tk: tk, model.global_step: step}
    return feed_dict

def evaluation(sess, model, data, user_idx, item_idx, text_dict, lam=None):
    p_implis = []
    p_explis = []
    for step in range(len(data)// model.bs+1):
        batch_outfits = data[step * model.bs:(step + 1) * model.bs]
        feed_dict = pocs_batch(model, batch_outfits, user_idx, item_idx, text_dict, 0)
        p_impli, p_expli = model.test(sess, feed_dict)
        for score_i in p_impli:
            p_implis.append(score_i)
        for score_i in p_expli:
            p_explis.append(score_i)

    p_implis = norm_array(np.reshape(np.array(p_implis), [-1, 101]))
    p_explis = norm_array(np.reshape(np.array(p_explis), [-1, 101]))

    if lam == None:
        max_auc_i = 0
        for lam in [0., 0.2, 0.4, 0.6, 0.8, 1.]:
            scores = lam * p_implis + (1 - lam) * p_explis
            AUC_i, MRR_i, HR10_i, N10 = metrics(scores, ['AUC', 'MRR', 'TOP@10', 'NDCG@10'])
            if AUC_i > max_auc_i:
                max_auc_i = AUC_i
                result_i=[AUC_i, MRR_i, HR10_i, N10, lam, p_implis, p_explis]
    else:
        # for lam in [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]:
        #     scores = lam * p_implis + (1 - lam) * p_explis
        #     AUC_i, MRR_i, HR10_i = metrics(scores, ['AUC', 'MRR', 'TOP@10'])
        #     print(lam, AUC_i, MRR_i, HR10_i)
        #     result_i = [AUC_i, MRR_i, HR10_i, lam]
        scores = lam * p_implis + (1 - lam) * p_explis
        AUC_i, MRR_i, HR10_i, N10 = metrics(scores, ['AUC', 'MRR', 'TOP@10', 'NDCG@10'])
        result_i = [AUC_i, MRR_i, HR10_i, N10, lam, p_implis, p_explis]
    return result_i

def metrics(array, metrics_map):
    out = []
    for metrics in metrics_map:
        if 'AUC' in metrics:
            a = np.ones([1])
            b = np.zeros([np.shape(array)[1]-1])
            label = np.concatenate([a, b])
            aa = []
            for i in array:
                aaa = sklearn.metrics.roc_auc_score(label, i)
                aa.append(aaa)
            score = np.average(aa)
            out.append(score)

        if 'MRR' in metrics:
            a = np.argsort(-array, axis=1)
            a = np.where(a == 0)[1]
            a = 1. / (a + 1)
            score = np.average(a)
            out.append(score)

        if 'TOP@' in metrics:
            k = int(metrics.replace('TOP@', ''))
            aaaa = []
            for i in array:
                aa = i - i[0]
                aa[aa > 0] = 1
                aa[aa < 0] = 0
                aaa = np.sum(aa)
                if aaa + 1 <= k:
                    aaaa.append(1)
                else:
                    aaaa.append(0)
            score = np.average(np.array(aaaa))
            out.append(score)

        if 'NDCG@' in metrics:
            k = int(metrics.replace('NDCG@', ''))
            a = np.ones([1], dtype=np.float32)
            b = np.zeros([np.shape(array)[1] - 1])
            label = np.concatenate([a, b])
            aa = []
            for i in array:
                aaa = sklearn.metrics.ndcg_score(y_true=[label], y_score=[i], k=k)
                aa.append(aaa)
            score = np.average(aa)
            out.append(score)

    return out
