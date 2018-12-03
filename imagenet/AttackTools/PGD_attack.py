#!/usr/bin/env python
# encoding: utf-8
'''
@author: lindq
@contact: lindq@shanghaitech.edu.cn
@time: 18-12-3 下午5:48
'''

import sys
import tensorflow as tf
import numpy as np

class PGDattack:
    def __init__(self, sess, model, batch_size=20):
        image_size, num_channels, num_labels = model.image_size, model.num_channels, model.num_labels
        self.sess = sess
        self.model = model

        shape = (batch_size, image_size, image_size, num_channels)

        self.image = tf.Variable(np.zeros(shape, dtype=np.float32))
        self.x = tf.placeholder(tf.float32, shape)
        # our trainable adversarial input
        self.x_hat = self.image
        self.assign_op = tf.assign(self.x_hat, self.x)

        self.learning_rate = tf.placeholder(tf.float32, ())
        self.y_hat = tf.placeholder(tf.int32, (batch_size, num_labels))

        self.logits, _ = self.model.predict(self.image)
        # labels = tf.one_hot(self.y_hat, 1000)
        # self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=[labels])
        self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=[self.y_hat])
        self.optim_step = tf.train.GradientDescentOptimizer(
            self.learning_rate).minimize(self.loss, var_list=[self.x_hat])
        self.epsilon = tf.placeholder(tf.float32, ())

        below = self.x - self.epsilon
        above = self.x + self.epsilon
        projected = tf.clip_by_value(tf.clip_by_value(self.x_hat, below, above), 0, 1)
        with tf.control_dependencies([projected]):
            self.project_step = tf.assign(self.x_hat, projected)

    def attack_batch(self, imgs, targets, steps=100, lr=1e-1, eps=2./255.,):
        # initialization step
        self.sess.run(self.assign_op, feed_dict={self.x: imgs})

        # projected gradient descent
        for i in range(steps):
            # gradient descent step
            _, loss_value = self.sess.run(
                [self.optim_step, self.loss],
                feed_dict={self.learning_rate: lr, self.y_hat: targets})
            # project step
            self.sess.run(self.project_step, feed_dict={self.x: imgs, self.epsilon: eps})
            if (i + 1) % 10 == 0:
                print('step %d, loss=%g' % (i + 1, loss_value.sum()))

        adv_imgs = self.x_hat.eval(session=self.sess)  # retrieve the adversarial example

        l_inf_dists = np.max(abs(adv_imgs - imgs), axis=(1,2,3))

        adv_labels = np.argmax(self.sess.run(self.logits, feed_dict={self.image: adv_imgs}), axis=1)
        return adv_imgs, adv_labels, l_inf_dists

if __name__ == '__main__':
    import sys
    import numpy as np
    model_name = 'base_inception_model'
    batch_size = 2
    sys.path.insert(0, '../')
    import stuff_v2 as stuff
    sess = stuff.get_session()
    m = stuff.M(model_name=model_name)

    a = PGDattack(sess=sess, model=m, batch_size=batch_size)
    img_path = '/media/line290/38284FF5284FB0A4/store/dataset/tinydataset/image/n02123045*281/ILSVRC2012_val_00007006.JPEG'
    nhwc = stuff.load_image(img_path=img_path)
    # nhwc = nhwc[np.newaxis,...]
    nhwc = np.stack((nhwc, nhwc))
    nhwc = nhwc / 255.
    print nhwc.shape
    labels = [847, 925]
    labels_one_hot = np.zeros((batch_size, m.num_labels), dtype=np.float32)
    labels_one_hot[range(batch_size), labels] = 1.
    adv_imgs, adv_label, l_inf_dist = a.attack_batch(nhwc, targets=labels_one_hot, steps=100, lr=1e-1, eps=2./255.)
    print adv_label, l_inf_dist, adv_imgs.shape