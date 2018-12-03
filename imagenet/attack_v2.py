
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tqdm
import scipy
# usage attack.py model_name attack_name offset count
import os
import sys
model_name, attack_name, offset, count = sys.argv[1:]
model_name = 'base_inception_model'
# model_name = 'adv_inception_v3'
offset = int(offset)
count = int(count)

import stuff_v2 as stuff
sess = stuff.get_session()
m = stuff.M(model_name)
# nhwc, labels = stuff.load_data()
# nhwc_slice = nhwc[offset:offset+count]
# labels_slice = labels[offset:offset+count]
batch_size = 1
from AttackTools import l2_attack_ensemble
confidences = 10
noise_count = 1
# a = l2_attack_ensemble.CarliniL2Ensemble(sess, m, batch_size=batch_size,
#                                          confidence=confidences, targeted=True,
#                                          learning_rate=1e-2, max_iterations=1000,
#                                          binary_search_steps=4, initial_const=1e-1,noise_count = noise_count)

# save_path = '/home/line290/Documents/project/ImageNet10_ensemble_l2_nmodel_1_confds_10/'
# save_path = '/home/line290/Documents/project/ImageNet10_ensemble_l2_nmodel_'+str(noise_count)+'_confds_'+str(confidences)+'_target/'
# save_path = '/media/line290/38284FF5284FB0A4/store/dataset/optens_20_confidence_0_img_1000/'
save_path = '/media/line290/38284FF5284FB0A4/store/dataset/tinydataset/'
if os.path.exists(save_path) == False:
    os.mkdir(save_path)
# img_path = '/home/line290/Documents/project/ImageNet10/matchstick_644'
img_path = '/media/line290/38284FF5284FB0A4/store/dataset/tinydataset/image/n02123045*281/ILSVRC2012_val_00007006.JPEG'
# img_list_path = '/media/line290/38284FF5284FB0A4/store/dataset/ImageNet1000.lst'
# img_list_path = '/home/line290/Documents/project/ImageNet10/ImageNet10_list.lst'
# # img_path = '/home/line290/Documents/project/ImageNet10_ensemble_l2_nmodel_20/matchstick_644_nmodel_20_l2dist_0.10583047_644'
# f = open(img_list_path)
# img_list = f.readlines()
ccccc = 0
# for img_path in img_list[0]:
for img_path in range(1):
    img_path = '/media/line290/38284FF5284FB0A4/store/dataset/tinydataset/image/n02123045*281/ILSVRC2012_val_00007006.JPEG'
    # if ccccc == 0:
    #     ccccc += 1
    #     continue
    # if ccccc == 0 or ccccc == 1:
    #     ccccc += 1
    #     continue
    # img_path = img_path.split('\n')[0]
    # nhwc, labels, img_name = stuff.load_image(img_path=img_path)
    nhwc = stuff.load_image(img_path=img_path)
    # nhwc, labels = stuff.load_img()
    nhwc_slice = nhwc
    # labels_slice = labels
    print(nhwc.shape)
    # print(nhwc[:10,])
    # print(labels_slice)
    # sess = stuff.get_session()
    # m = stuff.M(model_name)

    if 'none' == attack_name:
        imgs = nhwc_slice.astype(np.float32) / 255.
        np.save('%s_%s_%d_%d.npy' % (model_name, attack_name, offset, count), imgs)
    elif 'opt' == attack_name:
        batch_size = 1
        from AttackTools import l2_attack
        a = l2_attack.CarliniL2(sess, m, batch_size=batch_size,
                                confidence=0, targeted=True,
                                learning_rate=1e-2, max_iterations=1000, binary_search_steps=4, initial_const=1e-1,
                                boxmin=0., boxmax=1.)
        nbhwc = nhwc_slice.reshape((-1, batch_size, m.image_size, m.image_size, m.num_channels))
        # labels_batches = labels_slice.reshape((-1, batch_size))
        for i in tqdm.trange(nbhwc.shape[0]):
            offset_batch = offset + i * batch_size
            imgs = nbhwc[i].astype(np.float32) / 255.
            # labels = labels_batches[i]
            labels = 848
            labels_one_hot = np.zeros((batch_size, m.num_labels), dtype=np.float32)
            labels_one_hot[range(batch_size), labels] = 1.
            adv_imgs, l2_dist = a.attack_batch(imgs, labels_one_hot)
            print(l2_dist)
            # adv_imgs = a.attack(imgs, labels_one_hot)
            # np.save('%s_%s_%d_%d.npy' % (model_name, attack_name, offset_batch, batch_size), adv_imgs)
    elif 'optens' == attack_name:
        nbhwc = nhwc_slice.reshape((-1, batch_size, m.image_size, m.image_size, m.num_channels))
        # labels_batches = labels_slice.reshape((-1, batch_size))
        for i in tqdm.trange(nbhwc.shape[0]):
            offset_batch = offset + i * batch_size
            imgs = nbhwc[i].astype(np.float32) / 255.
            # labels = labels_batches[i]

            # labels = 935 # hotdog
            labels = 848 # guacamole
            print(labels)
            labels_one_hot = np.zeros((batch_size, m.num_labels), dtype=np.float32)
            labels_one_hot[range(batch_size), labels] = 1.
            print(labels_one_hot.shape)
            adv_imgs, l2_dist, pl = a.attack_batch(imgs, labels_one_hot)
            # adv_imgs, l2_dist = a.attack(imgs, labels_one_hot)
            print(adv_imgs[0].shape, l2_dist, pl)
            diff = adv_imgs[0] - imgs
            np.save(save_path+'diff.npy', diff)
            # x = tf.placeholder(shape=[None, m.image_size, m.image_size, m.num_channels], dtype=tf.float32)
            # y_pred = tf.argmax(m.predict(x), 1)
            # y_pred = sess.run(y_pred, feed_dict={x: adv_imgs[np.newaxis, ::]})
            # np.save(save_path+img_name+'_adv_'+str(pl)+'_l2_'+str(l2_dist)+'.npy', adv_imgs[0])
            scipy.misc.imsave(save_path+'ens_0_conf_100_GaussianBlur_adv_'+str(pl)+'_l2_'+str(l2_dist)+'.png', adv_imgs[0])
            # np.save('%s_%s/parts/%d_%d.npy' % (model_name, attack_name, offset_batch, batch_size), adv_imgs)
            # np.save('%s_%s/parts/%s_%d.npy' % (model_name, attack_name, img_name, batch_size), adv_imgs)
            # diff = adv_imgs[0] - imgs
            # np.save(save_path + 'diff_'+img_name + '_adv_' + str(pl) + '_l2_' + str(l2_dist) + '.npy', diff)
    elif 'fgsm' == attack_name:
        epsilon = 8. / 255.
        batch_size = 1
        imgs_ = np.load('hook, claw_600_adv_422_l2_[0.54431313].npy')
        print(imgs_)

        x = tf.placeholder(shape=[None, m.image_size, m.image_size, m.num_channels], dtype=tf.float32)
        y = tf.placeholder(shape=[None], dtype=tf.uint16)
        logits = m.predict(x)
        y_pred = tf.argmax(logits, 1)

        xent = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf.cast(y, tf.int32))
        xent_tot = tf.reduce_sum(xent)
        grad, = tf.gradients(xent_tot, x)
        x_adv = tf.clip_by_value(x + epsilon * tf.sign(grad), 0., 1.)

        nbhwc = nhwc_slice.reshape((-1, batch_size, m.image_size, m.image_size, m.num_channels))
        labels_batches = labels_slice.reshape((-1, batch_size))
        buf = np.zeros(nhwc_slice.shape, dtype=np.float32)
        buf_batches = buf.reshape(nbhwc.shape)
        for i in tqdm.trange(nbhwc.shape[0]):
            offset_batch = offset + i * batch_size
            imgs = nbhwc[i].astype(np.float32) / 255.
            print(np.sum(imgs_ - imgs))
            labels = labels_batches[i]
            buf_batches[i] = sess.run(x_adv, feed_dict={x: imgs, y: labels})
            print(sess.run(y_pred, feed_dict={x: imgs_[np.newaxis,::]}))
        # np.save('%s_%s_%d_%d.npy' % (model_name, attack_name, offset, count), buf)
    else:
        raise Exception('unknown attack')

    break