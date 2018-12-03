#!/usr/bin/env python
# encoding: utf-8
'''
@author: lindq
@contact: lindq@shanghaitech.edu.cn
@time: 18-11-5 上午11:37
'''
import os
import sys
import numpy as np
import tensorflow as tf
import PIL.Image
import matplotlib.pyplot as plt
import pickle
batchsize = 2

modelname = 'base_inception_model'
# modelname = 'adv_inception_v3'
# modelname = 'inception_resnet_v2'
modelname = 'resnet_v2_50'
# build model
import stuff_v2 as stuff
sess = stuff.get_session()
m = stuff.M(modelname)

x = tf.placeholder(shape=[batchsize, m.image_size, m.image_size, m.num_channels], dtype=np.float32)

logits, end_points = m.predict(x)
preds = tf.argmax(logits, axis=1) - 1

# each layer's name
# inceptionV3
end_points_key = ['input', 'Conv2d_1a_3x3', 'Conv2d_2a_3x3', 'Conv2d_2b_3x3',
      'MaxPool_3a_3x3', 'Conv2d_3b_1x1', 'Conv2d_4a_3x3', 'MaxPool_5a_3x3',
      'Mixed_5b', 'Mixed_5c', 'Mixed_5d', 'Mixed_6a', 'Mixed_6b', 'Mixed_6c',
      'Mixed_6d', 'Mixed_6e', 'Mixed_7a', 'Mixed_7b', 'Mixed_7c', 'logits']
# inception_resnet_v2
# end_points_key = ['input', 'Conv2d_1a_3x3', 'Conv2d_2a_3x3', 'Conv2d_2b_3x3',
#       'MaxPool_3a_3x3', 'Conv2d_3b_1x1', 'Conv2d_4a_3x3', 'MaxPool_5a_3x3',
#       'Mixed_5b', 'Mixed_6a', 'PreAuxLogits', 'Mixed_7a', 'Conv2d_7b_1x1', 'logits']


# the tuple of orig img and adv. img
sali_mp_prefix = 'smooth_guided.png'
attack1_prefix = 'target_ens_1_cfds_10'
# attack1_prefix = 'PGD_eps_8'
# attack1_prefix = 'target_ens_1_cfds_0'
orig_prefix = 'orig'
orig_prefix = attack1_prefix

# attack1_prefix = orig_prefix
prefix = '/media/line290/38284FF5284FB0A4/store/dataset/ImageNet20'

# init
results = {}
for key in end_points_key:
    results[key] = []

# prepare pair of imgs and save the L2 distances for each layer
folder_list = os.listdir(prefix)
if 1:
    for folder_name in folder_list:
        print folder_name
        img_folder_path = os.path.join(prefix, folder_name)
        img_list = os.listdir(img_folder_path)
        for img_full_name in img_list:
            if img_full_name[-len(sali_mp_prefix):] == sali_mp_prefix or img_full_name[-3:] == 'npy':
                continue
            if img_full_name[:len(attack1_prefix)] == attack1_prefix:
                adv_img_path = os.path.join(img_folder_path, img_full_name)
                adv_img = np.asarray(PIL.Image.open(adv_img_path)).astype(np.float32) /255.
                adv_img += np.random.uniform(-0.01,0.01, 299*299*3).reshape(299,299,3)
            if img_full_name[:len(orig_prefix)] == orig_prefix:
                orig_img_path = os.path.join(img_folder_path, img_full_name)
                orig_img = np.asarray(PIL.Image.open(orig_img_path)).astype(np.float32) / 255.
                # adv_img += np.random.uniform(-0.05, 0.05, 299 * 299 * 3).reshape(299, 299, 3)
        pair_img = np.stack([orig_img, adv_img], axis=0)
        # print pair_img.shape
        # break
        # diff of img

        # img_dist = np.sum((orig_img - adv_img)**2*4)/np.prod(orig_img.shape)

        # img_dist = np.sum((orig_img - adv_img)**2*4) / np.sum((orig_img - 0.5)**2*4)

        # img_dist = np.sum((orig_img - adv_img)**2*4)

        img_dist = np.sum(abs(orig_img - adv_img))*2 / np.sum(abs(orig_img))*2

        print folder_name, img_dist, np.prod(orig_img.shape)
        results[end_points_key[0]].append(img_dist)
        # print end_points_key[0], results[end_points_key[0]]
        pred_labels, all_outs, pred_logits = sess.run([preds,end_points, logits], feed_dict={x: pair_img})
        for key in all_outs.keys():
            print key
        break
        logits_dist = np.sum((pred_logits[0] - pred_logits[1])**2) / np.prod(pred_logits[0].shape)
        logits_dist = np.sum(abs((pred_logits[0] - pred_logits[1]))) / np.sum(abs(pred_logits[0]))


        results[end_points_key[-1]].append(logits_dist)
        for key in end_points_key[1:-1]:
            # orig_inter_output, adv_inter_output = all_outs[key][0], all_outs[key][1]
            # l2_dist = np.sum((all_outs[key][0] - all_outs[key][1])**2)

            # l2_dist = np.sum((all_outs[key][0] - all_outs[key][1])**2) / np.sum(all_outs[key][0]**2)

            # l2_dist = np.sum((all_outs[key][0] - all_outs[key][1])**2) / np.prod(all_outs[key][0].shape)

            l2_dist = np.sum(abs(all_outs[key][0] - all_outs[key][1])) / np.sum(abs(all_outs[key][0]))
            results[key].append(l2_dist)
            # print np.prod(all_outs[key][0].shape)
            print key, all_outs[key].shape, l2_dist
        print end_points_key[-1], pred_logits.shape, logits_dist
        print pred_labels
        # for out in all_outs:
        #     print out, all_outs[out].shape
    # if attack1_prefix == orig_prefix:
    #     handle = open(attack1_prefix+'_random_noise_001_l2_dist.pickle', 'wb')
    # else:
    #     handle = open(attack1_prefix + 'norm_l2_dist.pickle', 'wb')
    # pickle.dump(results, handle)
else:
    with open('orig_random_noise_001_l2_dist.pickle', 'rb') as handle:
        results = pickle.load(handle)
    with open(attack1_prefix+'_random_noise_001_l2_dist.pickle', 'rb') as handle:
    # with open('PGD_eps_8_l2_dist.pickle', 'rb') as handle:
        results1 = pickle.load(handle)
# labels = []
# idx = 14
# for i, folder_name in enumerate(folder_list):
#     rec = []
#     # if i < idx:
#     #     continue
#     # rec1 = []
#     for key in end_points_key:
#         rec.append(results[key][i])
#         # rec1.append(results1[key][i])
#         # print results[key]
#     # plt.plot(range(len(end_points_key)), np.log(np.asarray(rec)))
#     plt.plot(range(len(end_points_key)), np.asarray(rec))
#     # plt.plot(range(len(end_points_key)), np.log(np.asarray(rec1)), ':')
#     labels.append(folder_name.split('*')[-1])
#     # break
#
# # if attack1_prefix == orig_prefix:
# #     for i, folder_name in enumerate(folder_list):
# #         # rec = []
# #         rec1 = []
# #         # if i < idx:
# #         #     continue
# #         for key in end_points_key:
# #             # rec.append(results[key][i])
# #             rec1.append(results1[key][i])
# #             # print results[key]
# #         # plt.plot(range(len(end_points_key)), np.log(np.asarray(rec)))
# #         plt.plot(range(len(end_points_key)), np.log(np.asarray(rec1)), ':')
# #         labels.append(folder_name.split('*')[-1])
#         # break
#
# plt.xticks(range(len(end_points_key)), end_points_key)
# plt.legend(labels, loc='upper left', )
# if attack1_prefix == orig_prefix:
#     # title_str = attack1_prefix+ ' random noise(0.01) l2 distance (log)'
#     title_str = attack1_prefix+ ' random noise(0.01) l1 distance'
# else:
#     # title_str = attack1_prefix + ' norm l2 distance'
#     title_str = attack1_prefix + ' norm l1 distance'
#
# plt.title(title_str)
# # plt.ylabel(r"$\frac{||f_{adv}^x - f_{clean}^x||_2^2}{||f_{clean}^x||_2^2}$")
# plt.ylabel(r"$\frac{||f_{adv}^x - f_{clean}^x||}{||f_{clean}^x||}$")
# # plt.ylabel(r"normalized l2 distance")
# plt.xlabel(modelname+" name of layers")
#
# # plt.savefig('/media/line290/38284FF5284FB0A4/store/dataset/plot/'+title_str+'.png')
# plt.show()
# # plt.imshow()