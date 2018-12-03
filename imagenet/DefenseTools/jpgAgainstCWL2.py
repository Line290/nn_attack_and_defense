#!/usr/bin/env python
# encoding: utf-8
'''
@author: lindq
@contact: lindq@shanghaitech.edu.cn
@software: PyCharm
@file: repeat_object.py
@time: 18-10-24 下午8:41
@desc:
'''

import os
import sys
import numpy as np
import tensorflow as tf
import PIL.Image
import io

model_name = 'base_inception_model'
gpu_id, quality, cwl2_floder_path = sys.argv[1:]
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
#k = int(k)

import stuff_v2 as stuff
sess = stuff.get_session()
m = stuff.M(model_name)

x = tf.placeholder(shape=[None, m.image_size, m.image_size, m.num_channels], dtype=tf.float32)
# y = tf.placeholder(shape=[None], dtype=tf.uint16)
logits = m.predict(x)
y_pred = tf.argmax(logits, 1)
# max_logits = tf.reduce_max(logits, 1)
# min_logits = tf.reduce_min(logits, 1)
# cwl2_floder_path = '/home/piaozx/lindq/dataset/ImageNet_val_PGDfolder.lst'
f = open(cwl2_floder_path, 'r')
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
# f = open('jpg_recode.txt', 'wb')

num_img = 0
num_recover = 0
sum_dist = 0
# total_max_logits = []
# total_min_logits = []
for i, img_folder in enumerate(PGD_floder_path_list):
    # get adv img path
    img_folder = img_folder[:-1]
    _ , adv_img_path = img_folder.split('&')
    adv_img_path = adv_img_path.replace('//', '/')
    adv_img_name = adv_img_path.split('/')[-1][:-4]
    # target_ens_1_cfds_0_ol_65_tl_862_adv_862_L2_0.231744
    img_info = adv_img_name.split('_')
    adv = int(img_info[-3])
    if adv == -2:
        continue
    ol = int(img_info[6])
    l2_dist = float(img_info[-1])
    # -> jpg
    img = png2jpg(adv_img_path) / 255.
    # img = np.asarray(PIL.Image.open(img_path)) / 255.
    pred_label = sess.run(y_pred, feed_dict={x:img[np.newaxis,::]})
    print(pred_label[0]-1, ol)
    if (pred_label[0]-1) == int(ol):
        num_recover += 1
    num_img += 1
    sum_dist += l2_dist
    # total_max_logits.append(temp_max_logit[0])
    # total_min_logits.append(temp_min_logit[0])
    #print(temp_max_logit, temp_min_logit)
print('Total images: ', num_img)
print('The number of image back to original label:', num_recover)
print('Acc: ', num_recover*1.0/num_img)
if k!=0:
    print('Average l2 distance: ', sum_dist/num_img)
# _max_logits = np.asarray(total_max_logits)
# _min_logits = np.asarray(total_min_logits)
# _max_logits = np.sort(_max_logits)
# _min_logits = np.sort(_min_logits)
# print('max logits: ', _max_logits[::-1][:20])
# print('min logits: ', _min_logits[:20])
# np.save('total_max_logits.npy', total_max_logits)
# np.save('total_min_logits.npy', total_min_logits)

