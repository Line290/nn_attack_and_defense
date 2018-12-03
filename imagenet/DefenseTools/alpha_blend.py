#!/usr/bin/env python
# encoding: utf-8
'''
@author: lindq
@contact: lindq@shanghaitech.edu.cn
@software: PyCharm
@file: alpha_blend.py
@time: 18-10-24 下午5:12
@desc:
'''
import numpy as np
import tensorflow as tf
import tqdm
import scipy.misc
# usage attack.py model_name attack_name offset count
import os
import sys
img_path1, img_path2, r1, r2 = sys.argv[1:]
model_name = 'base_inception_model'

import stuff_v2 as stuff
sess = stuff.get_session()
m = stuff.M(model_name)

save_path = '/home/line290/Documents/project/ImageNet10_alpha_blend/'
# save_path = '/media/line290/38284FF5284FB0A4/store/dataset/optens_20_confidence_0_img_1000/'
if os.path.exists(save_path) == False:
    os.mkdir(save_path)
# img_path1 = '/home/line290/Documents/project/ImageNet10/matchstick_644'
# img_path2 = '/home/line290/Documents/project/ImageNet10/chow, chow chow_260'
img_name1 = img_path1.split('/')[-1]
img_name2 = img_path2.split('/')[-1]
# img_folder = img_path1[:-len(img_name1)]
im = stuff.alpha_blend_2_img(img_path1, img_path2, ratio1=float(r1), ratio2=float(r2))
scipy.misc.imsave(save_path+img_name1+'_'+img_name2+'.png', im)
x = tf.placeholder(shape=[None, m.image_size, m.image_size, m.num_channels], dtype=tf.float32)
logits = m.predict(x)
# y_pred = tf.argmax(logits, 1)
y_pred =  tf.nn.top_k(logits, k=5, sorted=True)
pred_label = sess.run(y_pred, feed_dict={x: im[np.newaxis,::]})
print pred_label
