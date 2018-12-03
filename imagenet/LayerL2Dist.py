#!/usr/bin/env python
# encoding: utf-8
'''
@author: lindq
@contact: lindq@shanghaitech.edu.cn
@time: 18-11-14
'''
import os
import sys
import numpy as np
import tensorflow as tf
import PIL.Image
import matplotlib.pyplot as plt
import pickle
batchsize = 4
tau = 0.01
modelname = 'base_inception_model'
# modelname = 'adv_inception_v3'
# modelname = 'inception_resnet_v2'
# modelname = 'resnet_v2_50'
# modelname = 'vgg_16'
# modelname = 'ens4_adv_inception_v3'
# build model
import stuff_v2 as stuff
sess = stuff.get_session()
m = stuff.M(modelname)

x = tf.placeholder(shape=[batchsize, m.image_size, m.image_size, m.num_channels], dtype=np.float32)

logits, end_points = m.predict(x)
preds = tf.argmax(logits, axis=1) - 1

# each layer's name
# inceptionV3
# end_points_key = ['input', 'Conv2d_1a_3x3', 'Conv2d_2a_3x3', 'Conv2d_2b_3x3',
end_points_key = ['Conv2d_1a_3x3', 'Conv2d_2a_3x3', 'Conv2d_2b_3x3',
      'MaxPool_3a_3x3', 'Conv2d_3b_1x1', 'Conv2d_4a_3x3', 'MaxPool_5a_3x3',
      'Mixed_5b', 'Mixed_5c', 'Mixed_5d', 'Mixed_6a', 'Mixed_6b', 'Mixed_6c',
      # 'Mixed_6d', 'Mixed_6e', 'Mixed_7a', 'Mixed_7b', 'Mixed_7c', 'logits']
      'Mixed_6d', 'Mixed_6e', 'Mixed_7a', 'Mixed_7b', 'Mixed_7c']
# inception_resnet_v2
# end_points_key = ['input', 'Conv2d_1a_3x3', 'Conv2d_2a_3x3', 'Conv2d_2b_3x3',
#       'MaxPool_3a_3x3', 'Conv2d_3b_1x1', 'Conv2d_4a_3x3', 'MaxPool_5a_3x3',
#       'Mixed_5b', 'Mixed_6a', 'PreAuxLogits', 'Mixed_7a', 'Conv2d_7b_1x1', 'logits']
# resnet50

# npy_cwl2_prefix = 'resnet50target_ens_1'
npy_cwl2_prefix = 'target_ens_1'
# npy_cwl2_prefix = 'target_ens_4'
# npy_cwl2_prefix = 'vgg16target_ens_1'

# the tuple of orig img and adv. img
sali_mp_prefix = 'smooth_guided.png'
# cwl2_attack_prefix = ['target_ens_4_cfds_0', 'target_ens_4_cfds_2', 'target_ens_4_cfds_5', 'target_ens_4_cfds_10', 'target_ens_4_cfds_20']
# cwl2_attack_prefix = ['vgg16target_ens_1_cfds_0', 'vgg16target_ens_1_cfds_2', 'vgg16target_ens_1_cfds_5', 'vgg16target_ens_1_cfds_10', 'vgg16target_ens_1_cfds_20']
cwl2_attack_prefix = ['target_ens_1_cfds_0', 'target_ens_1_cfds_2', 'target_ens_1_cfds_5', 'target_ens_1_cfds_10', 'target_ens_1_cfds_20']
# cwl2_attack_prefix = ['resnet50target_ens_1_cfds_0', 'resnet50target_ens_1_cfds_2', 'resnet50target_ens_1_cfds_5', 'resnet50target_ens_1_cfds_10', 'resnet50target_ens_1_cfds_20']

# pgd_attack_prefix = ['ens4InceptionV3PGD_eps_2', 'ens4InceptionV3PGD_eps_8', 'ens4InceptionV3PGD_eps_26', 'ens4InceptionV3PGD_eps_254']
# pgd_attack_prefix = ['vgg16PGD_eps_2', 'vgg16PGD_eps_8', 'vgg16PGD_eps_26', 'vgg16PGD_eps_254']
pgd_attack_prefix = ['PGD_eps_2', 'PGD_eps_8', 'PGD_eps_26', 'PGD_eps_254']
# pgd_attack_prefix = ['resnet50PGD_eps_2', 'resnet50PGD_eps_8', 'resnet50PGD_eps_26', 'resnet50PGD_eps_254']

# fgsm_attack_prefix = ['ens4_adv_inception_v3untarget_fgsm_1_eps_1', 'ens4_adv_inception_v3untarget_fgsm_1_eps_2', 'ens4_adv_inception_v3untarget_fgsm_1_eps_8', 'ens4_adv_inception_v3untarget_fgsm_1_eps_26']
# fgsm_attack_prefix = ['vgg_16untarget_fgsm_1_eps_1', 'vgg_16untarget_fgsm_1_eps_2', 'vgg_16untarget_fgsm_1_eps_8', 'vgg_16untarget_fgsm_1_eps_26']
fgsm_attack_prefix = ['untarget_fgsm_1_eps_1', 'untarget_fgsm_1_eps_2', 'untarget_fgsm_1_eps_8', 'untarget_fgsm_1_eps_26']
# fgsm_attack_prefix = ['resnet_v2_50untarget_fgsm_1_eps_1', 'resnet_v2_50untarget_fgsm_1_eps_2', 'resnet_v2_50untarget_fgsm_1_eps_8', 'resnet_v2_50untarget_fgsm_1_eps_26']
# attack1_prefix = 'target_ens_1_cfds_0'
orig_prefix = 'orig'
# orig_prefix = attack1_prefix
attack1_prefixs = cwl2_attack_prefix + pgd_attack_prefix + fgsm_attack_prefix
# attack1_prefix = orig_prefix
prefix = '/media/line290/38284FF5284FB0A4/store/dataset/ImageNet20'
# save_prefix = '/media/line290/38284FF5284FB0A4/store/dataset/ImageNet20_plot_resnet_tau_001'
# save_prefix = '/media/line290/38284FF5284FB0A4/store/dataset/ImageNet20_plot_inceptionV3_tau_001'
# save_prefix = '/media/line290/38284FF5284FB0A4/store/dataset/ImageNet20_plot_ens4_tau_01'
# save_prefix = '/media/line290/38284FF5284FB0A4/store/dataset/ImageNet20_plot_vgg16_tau_01'
if os.path.exists(save_prefix) == False:
    os.mkdir(save_prefix)
# init
results = {}
for key in end_points_key:
    results[key] = []

# prepare pair of imgs and save the L2 distances for each layer
folder_list = os.listdir(prefix)
if 1:
    # for folder_name in folder_list[1:]:
    for folder_name in folder_list[0:]:
        print folder_name
        img_folder_path = os.path.join(prefix, folder_name)
        img_list = os.listdir(img_folder_path)
        for attack1_prefix in attack1_prefixs:
            for img_full_name in img_list:
                if img_full_name[-len(sali_mp_prefix):] == sali_mp_prefix or (img_full_name[:len(npy_cwl2_prefix)] == npy_cwl2_prefix and img_full_name[-3:] == 'png'):
                    continue
                if img_full_name[:len(attack1_prefix)] == attack1_prefix:
                    adv_img_path = os.path.join(img_folder_path, img_full_name)
                    if img_full_name[:len(npy_cwl2_prefix)] != npy_cwl2_prefix:
                        adv_img = np.asarray(PIL.Image.open(adv_img_path)).astype(np.float32) /255.
                    else:
                        adv_img = np.load(adv_img_path).astype(np.float32)

                    # noise = np.zeros(shape=299*299*3, dtype=np.float32)
                    # idx = np.random.randint(0, 299*299*3, size=10000)
                    # # for i in range(len(idx)):
                    # noise[idx] = np.random.uniform(-tau, tau, len(idx))
                    # adv_img_r = adv_img + noise.reshape(299,299,3)

                    adv_img_r = adv_img + np.random.uniform(-tau, tau, 299*299*3).reshape(299,299,3)
                if img_full_name[:len(orig_prefix)] == orig_prefix:
                    orig_img_path = os.path.join(img_folder_path, img_full_name)
                    orig_img = np.asarray(PIL.Image.open(orig_img_path)).astype(np.float32) / 255.

                    # noise = np.zeros(shape=299 * 299 * 3, dtype=np.float32)
                    # idx = np.random.randint(0, 299 * 299 * 3, size=10000)
                    # # for i in range(len(idx)):
                    # noise[idx] = np.random.uniform(-tau, tau, len(idx))
                    # orig_img_r = orig_img + noise.reshape(299, 299, 3)

                    orig_img_r = orig_img + np.random.uniform(-tau, tau, 299 * 299 * 3).reshape(299, 299, 3)
            pair_img = np.stack([orig_img, adv_img, adv_img_r, orig_img_r], axis=0)

            pred_labels, all_outs, pred_logits = sess.run([preds, end_points, logits], feed_dict={x: pair_img})
            print pred_labels



            # results[end_points_key[-1]].append(logits_dist)
            xr_x, xpr_xp, xpr_x = [], [], []

            # for key in end_points_key:
             # resnet_v2_50 has softmax layer
            # end_points_key = all_outs.keys()[:-1]
            end_points_key = all_outs.keys()
            for key in end_points_key:
                # x, xp, xpr, xr
                # xr_x, xpr_xp, xpr_x

                # temp_xr_x = np.sum(abs(all_outs[key][3] - all_outs[key][0])) / np.sum(abs(all_outs[key][0]))
                # temp_xpr_xp = np.sum(abs(all_outs[key][2] - all_outs[key][1])) / np.sum(abs(all_outs[key][1]))
                # temp_xpr_x = np.sum(abs(all_outs[key][2] - all_outs[key][0])) / np.sum(abs(all_outs[key][0]))

                temp_xr_x = np.sum((all_outs[key][3] - all_outs[key][0])**2) / np.prod(all_outs[key][0].shape)
                temp_xpr_xp = np.sum((all_outs[key][2] - all_outs[key][1])**2) / np.prod(all_outs[key][1].shape)
                temp_xpr_x = np.sum((all_outs[key][2] - all_outs[key][0])**2) / np.prod(all_outs[key][0].shape)

                # temp_xpr_x = np.sum((all_outs[key][1] - all_outs[key][0])**2 / np.prod(all_outs[key][0].shape))
                print key, all_outs[key].shape, temp_xpr_x
                xr_x.append(temp_xr_x)
                xpr_xp.append(temp_xpr_xp)
                xpr_x.append(temp_xpr_x)
                # print key
            plt.figure(figsize=(30,15))
            # plt.plot(range(len(end_points_key)), np.log(np.asarray(xr_x)), '-')
            # plt.plot(range(len(end_points_key)), np.log(np.asarray(xpr_xp)), '-.')
            # plt.plot(range(len(end_points_key)), np.log(np.asarray(xpr_x)), ':')

            plt.plot(range(len(end_points_key)), np.asarray(xr_x), '-')
            plt.plot(range(len(end_points_key)), np.asarray(xpr_xp), '-.')
            plt.plot(range(len(end_points_key)), np.asarray(xpr_x), ':')

            plt.xticks(range(len(end_points_key)), end_points_key)
            plt.legend(['xr_x', 'xpr_xp', 'xpr_x'], loc='upper left', )
            # plt.legend(['xr_x', 'xpr_xp'], loc='upper left', )
            title_str = attack1_prefix + '_' + folder_name + '_tau_'+str(tau)
            plt.title(title_str+'_[x,xp,xpr,xr]_'+str(pred_labels))
            # plt.ylabel(r"$\frac{||f_{adv}^x - f_{clean}^x||_2^2}{||f_{clean}^x||_2^2}$")
            # plt.ylabel(r"$\frac{||f_{adv}^x - f_{clean}^x||}{||f_{clean}^x||}$")
            plt.ylabel(r"log of normalized l2 distance")
            plt.xlabel(modelname + " name of layers")
            save_folder = os.path.join(save_prefix, folder_name)
            if os.path.exists(save_folder) == False:
                os.mkdir(save_folder)
            # plt.savefig(os.path.join(save_folder, title_str+'.png'), bbox_inches='tight')
            plt.show()
# else:
#     with open('orig_random_noise_001_l2_dist.pickle', 'rb') as handle:
#         results = pickle.load(handle)
#     with open(attack1_prefix+'_random_noise_001_l2_dist.pickle', 'rb') as handle:
#     # with open('PGD_eps_8_l2_dist.pickle', 'rb') as handle:
#         results1 = pickle.load(handle)
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

# plt.savefig('/media/line290/38284FF5284FB0A4/store/dataset/plot/'+title_str+'.png')
# plt.show()
# plt.imshow()