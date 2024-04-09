import tensorflow as tf
import tensorflow.contrib as tc
import numpy as np
from helper import *
import os
import random

""" DUPLE
Reference:
    "Dual Preference Distribution Learning for Item Recommendation"
    Dong et al., TOIS'2023.
CMD example:
    python3 DUPLE.py --dataset ml-small
"""

class DUPLE(object):
    def __init__(self, dataset, bs, hidden_dim, d, voc):
        self.d = d
        self.bs = bs
        self.voc_size = len(voc)
        self.hidden_dim = hidden_dim
        self.voc = voc
        self.lr = 0.0005
        self.initializer = tf.random_normal_initializer(stddev=0.1)
        if dataset in ['men','women','phone']:
            self.fea_dim = 4096
        elif dataset in ['ml-small','ml-1m','ml-10m']:
            self.fea_dim = len(self.voc)
    
    def extractor(self, feature, name, reuse=False):
        with tf.variable_scope(name) as scope:
            if reuse:
                scope.reuse_variables()
            h1 = tf.nn.relu(linear(feature, self.hidden_dim, name='h1'))
            h2 = linear(h1, self.hidden_dim, name='h2')
            h2_norm = tf.nn.l2_normalize(h2, 1)
        return h2_norm

    def learning_G(self, feature):
        mu = linear(feature, self.hidden_dim, name='mu')
        mu = tf.nn.l2_normalize(mu, 1)
        sigma_V = []
        for i in range(self.d):
            a_i = tf.nn.relu(linear(feature, self.hidden_dim, name='sigma%d'%i)) # bs, fea_dim
            sigma_V.append(a_i)
        sigma_V = tf.reshape(tf.concat(sigma_V, 1), [-1, self.d, self.hidden_dim])
        sigma = tf.matmul(tf.transpose(sigma_V, [0, 2, 1]), sigma_V)
        return mu, sigma, sigma_V

    def transformation(self, feature, reuse=False):
        with tf.variable_scope('transformation') as scope:
            if reuse:
                scope.reuse_variables()
            shape_ = feature.get_shape().as_list()
            if len(shape_) == 2:
                feature = linear(feature, self.hidden_dim, name='h1', if_bias=False)
                feature = linear(feature, self.voc_size, name='h2')
                feature = tf.nn.l2_normalize(feature, 1)
            else:
                feature = tf.reshape(feature, [-1, self.hidden_dim])
                feature = linear(feature, self.hidden_dim, name='h1', if_bias=False)
                feature = linear(feature, self.voc_size, name='h2')
                feature = tf.reshape(feature, [-1, self.d, self.voc_size])
                feature = tf.matmul(tf.transpose(feature, [0, 2, 1]), feature)
        return feature

    def pdt_loss(self, pos, neg, anchor, sigma):
        _, dim = pos.get_shape().as_list()
        anchor = tf.reshape(anchor, [-1, 1, dim])
        pos = tf.reshape(pos, [-1, 1, dim])
        neg = tf.reshape(neg, [-1, 1, dim])
        pos_score = -tf.matmul(tf.matmul((anchor - pos), sigma), (anchor - pos), transpose_b=True)
        neg_score = -tf.matmul(tf.matmul((anchor - neg), sigma), (anchor - neg), transpose_b=True)
        pos_score = tf.squeeze(pos_score)
        neg_score = tf.squeeze(neg_score)
        loss = tf.reduce_mean(-tf.log(tf.exp(pos_score) / (tf.exp(pos_score) + tf.exp(neg_score))))
        return loss, pos_score

    def consist_loss(self, pos, neg, anchor):
        pos_score = tf.reduce_sum(tf.multiply(pos, anchor), 1)
        neg_score = tf.reduce_sum(tf.multiply(neg, anchor), 1)
        loss = tf.reduce_mean(-tf.log(tf.exp(pos_score) / (tf.exp(pos_score) + tf.exp(neg_score))))
        return loss, pos_score

    def model(self):
        self.u = tf.placeholder(shape=[None], dtype=tf.int32, name='user')
        self.i = tf.placeholder(shape=[None], dtype=tf.int32, name='item')
        self.k = tf.placeholder(shape=[None], dtype=tf.int32, name='itemk')
        self.fi = tf.placeholder(shape=[None, self.fea_dim], dtype=tf.float32, name='fi')
        self.fk = tf.placeholder(shape=[None, self.fea_dim], dtype=tf.float32, name='fk')
        self.ti = tf.placeholder(shape=[None, self.voc_size], dtype=tf.float32, name='ti')
        self.tk = tf.placeholder(shape=[None, self.voc_size], dtype=tf.float32, name='tk')

        I = tf.get_variable('I', shape=[len(item_idx), self.hidden_dim], initializer=self.initializer)
        U = tf.get_variable('U', shape=[len(user_idx), self.hidden_dim], initializer=self.initializer)
        u_gather = tf.gather(U, self.u)
        i_gather = tf.gather(I, self.i)
        k_gather = tf.gather(I, self.k)

        fu = self.extractor(u_gather, 'user')
        fi = self.extractor(i_gather, 'item')
        fk = self.extractor(k_gather, 'item', True)

        ## general interest learning
        self.mu, self.sigma, sigma_V = self.learning_G(fu)

        ## specific interest learning
        ti_hat = self.transformation(fi)
        ti = tf.nn.l2_normalize(self.ti, 1)
        tk = tf.nn.l2_normalize(self.tk, 1)
        self.mu_hat = self.transformation(self.mu, reuse=True)
        self.sigma_hat = self.transformation(sigma_V, reuse=True)

        ## define loss
        self.loss_impli, self.p_impli = self.pdt_loss(fi, fk, self.mu, self.sigma)
        self.loss_expli, self.p_expli = self.pdt_loss(ti, tk, self.mu_hat, self.sigma_hat)
        self.loss_consis, _ = self.consist_loss(ti, tk, ti_hat)
        self.loss = self.loss_impli + self.loss_expli + self.loss_consis

        self.reg_loss = tc.layers.apply_regularization(tc.layers.l1_regularizer(0.00001),
                                                  weights_list=[var for var in tf.global_variables() if
                                                                'kernel' in var.name])
        ## optim
        self.global_step = tf.Variable(0, dtype=tf.int64, name='global_step', trainable=False)
        self.optim = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        # selective addition of regularization loss
        # self.optim = tf.train.AdamOptimizer(self.lr).minimize(self.loss + self.reg_loss)

    def train(self, sess, feed_dict):
        fetch = sess.run([self.optim, self.loss_impli, self.loss_expli, self.loss_consis, self.reg_loss], feed_dict)
        return fetch

    def test(self, sess, feed_dict):
        fetch = sess.run([self.p_impli, self.p_expli], feed_dict)
        return fetch

    def save(self, saver, sess, save_path, step):
        saver.save(sess, save_path, global_step=step)


## load_data
parser = argparse.ArgumentParser()
parser.add_argument('--dataset',default='men')
parser.add_argument('--bs',default=512)
parser.add_argument('--epoch',default=100,type=int)
parser.add_argument('--hidden_dim',default=64)
parser.add_argument('--lam',default=None,type=float)
parser.add_argument('--d',default=8,type=int)
args = parser.parse_args()

args.dataset = args.dataset.replace('\r', '')

data_path,item_data,train_data,valid_data,test_data,user_idx,item_idx,voc,text_dict = data_loader(args.dataset)
model = DUPLE(dataset=args.dataset, bs=args.bs, hidden_dim=args.hidden_dim, d=args.d, voc=voc)
model.model()


## train
checkpoint_path = './checkpoints/%s/' % args.dataset
max_auc = 0
max_step = 0
config = tf.ConfigProto()
sess = tf.Session(config=config)
saver = tf.train.Saver(max_to_keep=1)
init_op = tf.global_variables_initializer()
sess.run(init_op)
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)

for epoch in range(int(args.epoch)):
    random.shuffle(train_data)
    all_step = len(train_data) // model.bs
    for step in range(all_step):
        global_step = epoch * all_step + step
        batch_outfits = train_data[step * model.bs:(step + 1) * model.bs]
        feed_dict = pocs_batch(model, batch_outfits, user_idx, item_idx, text_dict, global_step)
        _, loss_dis, p_expli, loss_consis, reg_loss = model.train(sess, feed_dict)

        ## validation
        if step % 1000 == 0:
            ## valid partly to speed up
            # AUC, MRR, HR10, N10, lam, _, _ = evaluation(sess, model, valid_data[:641*101], user_idx, item_idx, text_dict, args.lam)
            AUC, MRR, HR10, N10, lam, _, _ = evaluation(sess, model, valid_data[:], user_idx, item_idx, text_dict, args.lam)
            if AUC > max_auc:
                saver.save(sess, checkpoint_path + '%.2f' % AUC, global_step=step)
                final_lam = lam
                max_auc = AUC
                max_step = global_step

            epoch_show = '%'+'0%sd'%len(str(args.epoch))
            step_show = '%'+'0%sd'%len(str(all_step))
            log = ['%s: [',epoch_show,'/%d][',step_show,'/%d] | loss: %.3f, %.3f, %.3f, %.3f | auc-mrr-hr-ndcg: %.4f-%.4f-%.4f-%.4f | Max: %.4f(%d-%.1f)']
            log = ''.join(log)
            log = log % (args.dataset, epoch, args.epoch, step, all_step, loss_dis, p_expli, loss_consis,
                         reg_loss, AUC, MRR, HR10, N10, max_auc, max_step, final_lam)
            print(log)
            log_file = open('./log_%s.txt'%args.dataset, 'a+')
            log_file.write(log+'\n')
            log_file.close()
sess.close()

config = tf.ConfigProto()
sess = tf.Session(config=config)
saver = tf.train.Saver(max_to_keep=1)

ckpt = tf.train.get_checkpoint_state(checkpoint_path)
saver.restore(sess, ckpt.model_checkpoint_path)

AUC, MRR, HR10, N10, lam, p_implis, p_explis = evaluation(sess, model, test_data, user_idx, item_idx, text_dict, args.lam)
log = 'Final Test Results for %s Dataset | auc-mrr-hr-ndcg: %.4f-%.4f-%.4f-%.4f |' % (args.dataset, AUC, MRR, HR10, N10)
print(log)
log_file = open('./log_%s.txt' % args.dataset, 'a+')
log_file.write(log + '\n')
log_file.close()






