#!/usr/bin/env python
# encoding: utf-8
'''
@author: lindq
@contact: lindq@shanghaitech.edu.cn
@software: PyCharm
@file: repeat_object.py
@time: 18-10-24
@desc:
'''

import os
import sys
import numpy as np
import tensorflow as tf
import PIL.Image
import io

model_name = 'base_inception_model'
k, gpu_id, quality = sys.argv[1:]
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
k = int(k)

import stuff_v2 as stuff
sess = stuff.get_session()
m = stuff.M(model_name)

x = tf.placeholder(shape=[None, m.image_size, m.image_size, m.num_channels], dtype=tf.float32)
# y = tf.placeholder(shape=[None], dtype=tf.uint16)
logits = m.predict(x)
y_pred = tf.argmax(logits, 1)

PGD_floder_path = '/home/piaozx/lindq/dataset/ImageNet_val_PGDfolder.lst'
f = open(PGD_floder_path, 'r')
PGD_floder_path_list = f.readlines()
f.close()

# def get_eps_path(folder_path, eps = 2):
#     img_name_list = os.listdir(folder_path)
#     for img_name in img_name_list:
#         if img_name.split('_')[2] == str(eps):
#             return os.path.join(folder_path, img_name)
#     return False
def png2jpg(img_path):
    d_img = io.BytesIO()
    img = PIL.Image.open(img_path)
    img.save(d_img, "JPEG", quality=int(quality))
    img = PIL.Image.open(d_img)
    return np.asarray(img)
f = open('jpg_recode.txt', 'wb')

num_img = 0
num_recover = 0
sum_dist = 0
for i, img_folder in enumerate(PGD_floder_path_list):
    img_name_list = os.listdir(img_folder[:-1])
    if len(img_name_list) != 5:
        continue
    orig_path = img_name_list[0]
    _, tl, _, pl = orig_path[:-4].split('_')
    if tl != pl:
        continue
    # for j in range(1,4):
    num_img += 1
    img_name = img_name_list[k]
    l2_dist = float(img_name[:-4].split('_')[-1])
    sum_dist += l2_dist
    img_path = os.path.join(img_folder[:-1], img_name)
    print(img_name_list[k])
    # -> jpg
    img = png2jpg(img_path) / 255.
    pred_label = sess.run(y_pred, feed_dict={x:img[np.newaxis,::]})
    print(pred_label[0]-1, tl)
    if (pred_label[0]-1) == int(tl):
        num_recover += 1
print('Total images: ', num_img)