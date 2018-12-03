#!/usr/bin/env python
# encoding: utf-8
'''
@author: lindq
@contact: lindq@shanghaitech.edu.cn
@time: 18-12-3 下午9:39
'''

from AttackTools.PGD_attack import PGDattack
from AttackTools.l2_attack import CarliniL2
from AttackTools.FGSM_attack import FGSM
if __name__ == '__main__':
    import sys
    import numpy as np

    model_name, attack_name = sys.argv[1:]
    model_name = 'base_inception_model'
    # attack_name = 'CWl2'
    batch_size = 2
    # sys.path.insert(0, '../')
    import stuff_v2 as stuff
    sess = stuff.get_session()
    m = stuff.M(model_name=model_name)

    # x: n*h*w*c
    img_path = '/media/line290/38284FF5284FB0A4/store/dataset/tinydataset/image/n02123045*281/ILSVRC2012_val_00007006.JPEG'
    nhwc = stuff.load_image(img_path=img_path)
    # nhwc = nhwc[np.newaxis,...]
    nhwc = np.stack((nhwc, nhwc, nhwc, nhwc))
    nhwc = nhwc / 255.
    print nhwc.shape
    # y: n
    labels = np.asarray([847, 925, 935, 229]).astype(np.int32)

    offset = 0
    count = 4
    # nhwc, labels = stuff.load_data()
    nhwc_slice = nhwc[offset:offset+count]
    labels_slice = labels[offset:offset+count]
    nbhwc = nhwc_slice.reshape((-1, batch_size, m.image_size, m.image_size, m.num_channels))
    labels_batches = labels_slice.reshape((-1, batch_size))

    if 'PGD' == attack_name:
        a = PGDattack(sess=sess, model=m, batch_size=batch_size)
        for i in range(nbhwc.shape[0]):
            batch_orig_imgs = nbhwc[i]
            batch_orig_labels = labels_batches[i]
            labels_one_hot = np.zeros((batch_size, m.num_labels), dtype=np.float32)
            labels_one_hot[range(batch_size), batch_orig_labels] = 1.

            adv_imgs, adv_label, l_inf_dist = a.attack_batch(batch_orig_imgs,
                                                             targets=labels_one_hot,
                                                             steps=100,
                                                             lr=1e-1,
                                                             eps=8./255.)
            print adv_label, l_inf_dist, adv_imgs.shape
    elif 'CWl2' == attack_name:
        a = CarliniL2(sess=sess,
                      model=m,
                      batch_size=batch_size,
                      confidence=0,
                      targeted=True,
                      learning_rate=1e-2,
                      max_iterations=1000,
                      binary_search_steps=4,
                      initial_const=1e-1,
                      boxmin=0.,
                      boxmax=1.)
        for i in range(nbhwc.shape[0]):
            batch_orig_imgs = nbhwc[i]
            batch_orig_labels = labels_batches[i]
            labels_one_hot = np.zeros((batch_size, m.num_labels), dtype=np.float32)
            labels_one_hot[range(batch_size), batch_orig_labels] = 1.

            adv_imgs, l2_dist = a.attack_batch(batch_orig_imgs, labels_one_hot)
            print adv_imgs.shape, l2_dist

    elif 'fgsm' == attack_name:
        a = FGSM(sess=sess,
                 model=m,
                 batch_size=batch_size)
        for i in range(nbhwc.shape[0]):
            batch_orig_imgs = nbhwc[i]
            batch_orig_labels = labels_batches[i]
            labels_one_hot = np.zeros((batch_size, m.num_labels), dtype=np.float32)
            labels_one_hot[range(batch_size), batch_orig_labels] = 1.

            adv_imgs, adv_labels = a.attack_batch(batch_orig_imgs, labels_one_hot, eps=8./255.)
            print adv_imgs.shape, adv_labels