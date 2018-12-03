#!/usr/bin/env python
# encoding: utf-8
'''
@author: lindq
@contact: lindq@shanghaitech.edu.cn
@software: PyCharm
@file: eval_vote.py.py
@time: 18-10-28 下午10:23
@desc:
'''
import os
import sys
import numpy as np
import tensorflow as tf
import tqdm
# import PIL.Image

imin = 0.
imax = 1.
ih = 299
iw = 299
ic = 3
ifeat = ih * iw * ic
num_classes = 1001

# cg_num_per = 100
# cg_tau = 0.02
cg_num_per = 100
# batchsize = 20
batchsize = 20
# cg_tau = 0.02
# cg_tau = 0.4
count = 1
# usage: python eval_cg.py <model> <dataset name> <images path> <labels path> <count>

cg_tau, GPU_ID = sys.argv[1:]
cg_tau = float(cg_tau)
modelname = 'base_inception_model'
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

import stuff_v2 as stuff
sess = stuff.get_session()
m = stuff.M(modelname)

x = tf.placeholder(shape=[ifeat], dtype=tf.float32)
perturbation = tf.random_uniform((batchsize, ifeat), minval=-cg_tau, maxval=cg_tau, dtype=tf.float32)
avg_rms = tf.reduce_mean(tf.sqrt(tf.reduce_mean(tf.square(perturbation), axis=1)))
# x_samples = tf.clip_by_value(x[None] + perturbation, imin, imax)
x_samples = x[None] + perturbation

input_tensor = tf.reshape(x_samples, [batchsize, ih, iw, ic])
logits = m.predict(tf.reshape(x_samples, [batchsize, ih, iw, ic]))

# preds = tf.argmax(logits, axis=1)
# pred_count = tf.unsorted_segment_sum(data=tf.ones(cg_num_per, dtype=tf.int32), segment_ids=preds, num_segments=num_classes)
from stuff_v2 import load_image
val_dataset_list_path = '/home/piaozx/lindq/dataset/ImageNet_val.lst'
f = open(val_dataset_list_path, 'r')
val_img_paths = f.readlines()
f.close()

rand1_count = 0
rand100_count = 0
total_count = 0
for val_img_path in val_img_paths:
    val_img_path = val_img_path[:-1]
    xs, ys = load_image(val_img_path)
    if xs.shape[2] == 3:
        continue
    if xs.shape[1] != ifeat:
        xs = xs.reshape((-1, ifeat))

    total_count += 1
    pred_counts_all = np.zeros((count, num_classes), dtype=np.int32)
    top_preds = np.zeros(count, dtype=np.uint16)
    rms_all = np.zeros(count, dtype=np.float32)
    for j in tqdm.trange(count, leave=False):
        # pred, pred_counts_all[j], rms_all[j] = sess.run([preds, pred_count, avg_rms], feed_dict={x: xs[j]})
        # top_preds[j] = np.argmax(pred_counts_all[j])
        for i in range(int(cg_num_per/batchsize)):
            temp_pred_logits = sess.run(logits, feed_dict={x:xs[j]})
            if i == 0:
                pred_logits = temp_pred_logits
            else:
                pred_logits = np.vstack((pred_logits, temp_pred_logits))
        preds = pred_logits.argmax(axis=1)
        top_logits_idx = pred_logits.argsort(axis=1)[:,::-1][:,:5]
        top_logits = pred_logits[range(preds.shape[0]),top_logits_idx]
        preds = np.append(preds, 1000)
        pred_counts_all[j] = np.bincount(preds)
        pred_counts_all[j][1000] -= 1
        top_preds[j] = np.argmax(pred_counts_all[j])
        if preds[j] == ys[j]:
            rand1_count += 1
        if top_preds[j] == ys[j]:
            rand100_count += 1
        print(val_img_path+'*r1*'+str(preds[j])+'*r100*'+str(top_preds[j]))
print("when the scale of uniform noise is [%.1f, %.1f]" %(-cg_tau, cg_tau))
print("Random 1 Acc : ", rand1_count/total_count)
print("Random 100 Acc : ", rand100_count/total_count)