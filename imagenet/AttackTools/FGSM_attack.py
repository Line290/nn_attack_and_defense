#!/usr/bin/env python
# encoding: utf-8
'''
@author: lindq
@contact: lindq@shanghaitech.edu.cn
@time: 18-12-3 下午10:35
'''

import sys
import tensorflow as tf
import numpy as np

class FGSM:
    def __init__(self, sess, model, batch_size=20):
        image_size, num_channels, num_labels = model.image_size, model.num_channels, model.num_labels
        self.sess = sess
        self.model = model

        shape = (batch_size, image_size, image_size, num_channels)
        self.x = tf.placeholder(shape=shape, dtype=tf.float32)
        self.y = tf.placeholder(tf.int32, (batch_size, num_labels))
        self.epsilon = tf.placeholder(tf.float32, ())
        logits, _ = self.model.predict(self.x)
        self.y_pred = tf.argmax(logits, 1)

        xent = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=[self.y])
        xent_tot = tf.reduce_sum(xent)
        grad = tf.gradients(xent_tot, self.x)[0]
        self.x_adv = tf.clip_by_value(self.x + self.epsilon * tf.sign(grad), 0., 1.)

    def attack_batch(self, imgs, labels, eps=2./255.):
        adv_imgs = self.sess.run(self.x_adv,
                                 feed_dict={self.x: imgs, self.y: labels, self.epsilon: eps})

        adv_labels = self.sess.run(self.y_pred,
                                   feed_dict={self.x: adv_imgs})
        return adv_imgs, adv_labels