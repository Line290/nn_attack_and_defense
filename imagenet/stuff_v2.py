from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import numpy as np
import tensorflow as tf
import scipy.misc
sess = None

def get_session():
    global sess
    if sess is None:
        sess = tf.Session()
    return sess

# ckpt_dir = './model_ckpt'
ckpt_dir = '/media/line290/38284FF5284FB0A4/store/model_ckpt'

def apply_model(x, net_fn, arg_scope_cls, ckpt_name, no_background=False, **kwargs):
    mv_start = len(tf.global_variables())
    arg_scope = arg_scope_cls()
    print('building model', net_fn.__name__,)
    sys.stdout.flush()
    with tf.contrib.slim.arg_scope(arg_scope):
        logits, end_points = net_fn(x, is_training=False, **kwargs)
        # logits = tf.reshape(logits, [-1,1001])
    print('done')
    mv_end = len(tf.global_variables())
    model_vars = tf.global_variables()[mv_start:mv_end]


    model_vars = [var for var in model_vars
                  if var.name.startswith('InceptionV3/InceptionV3/smoothing')==False and
                  var.name.startswith('InceptionV3/')]

    saver = tf.train.Saver(model_vars)

    # for v in tf.global_variables(): # %%%
    #     print(v.name)
    # exit(1) # %%%
    print('restoring model', ckpt_name,)
    sys.stdout.flush()
    saver.restore(sess, ckpt_dir + '/' + ckpt_name)
    print('done')
    print('initalize gaussian var')
    uninitalized_vars = []
    for var in tf.global_variables():
        try:
            sess.run(var)
        except tf.errors.FailedPreconditionError:
            uninitalized_vars.append(var)
    init_new_vars_op = tf.variables_initializer(uninitalized_vars)
    sess.run(init_new_vars_op)
    if no_background:
        logits = tf.concat([tf.fill([tf.shape(logits)[0], 1], -1e5), logits], axis=1)
    logits = logits[:,1:]
    return logits, end_points
    # return logits

class M:
    def __init__(self, model_name):
        self.model_name = model_name
        self.image_size = 299
        self.num_channels = 3
        self.num_labels = 1000
    def predict(self, x):
        x_inception = x * 2 - 1
        x_224 = tf.image.resize_bilinear(x, [224, 244])
        # x_inception_224 = x_224 * 2 - 1
        VGG_NHWC_MEAN = [123.68, 116.78, 103.94]
        x_vgg = x_224 * 255 - VGG_NHWC_MEAN

        if 'inception_resnet_v2' == self.model_name:

            # from tensorflow.contrib.slim.nets import inception_resnet_v2
            # sys.path.insert(0, '/usr/local/lib/python2.7/dist-packages/tensorflow/models/research/slim/')
            # for p in sys.path:
            #     print(p)
            # from models.research.slim.nets import inception_resnet_v2
            from nets import inception_resnet_v2
            return apply_model(x_inception,
                               inception_resnet_v2.inception_resnet_v2,
                               inception_resnet_v2.inception_resnet_v2_arg_scope,
                               'inception_resnet_v2_2016_08_30.ckpt')

        if 'inception_v4' == self.model_name:
            from nets import inception_v4
            return apply_model(x_inception,
                               inception_v4.inception_v4,
                               inception_v4.inception_v4_arg_scope,
                               'inception_v4.ckpt')

        # if 'mobilenet_v1' == self.model_name:
        #     from nets import mobilenet_v1
        #     return apply_model(x_inception_224,
        #                        mobilenet_v1.mobilenet_v1,
        #                        mobilenet_v1.mobilenet_v1_arg_scope,
        #                        'mobilenet_v1_1.0_224.ckpt',
        #                        num_classes=1001)

        if 'resnet_v2_50' == self.model_name:
            #raise Exception('broken. joystick or binoculars')
            from nets import resnet_v2
            # from tensorflow.contrib.slim.nets import resnet_v2
            return apply_model(x_inception,
                               resnet_v2.resnet_v2_50,
                               resnet_v2.resnet_arg_scope,
                               'resnet_v2_50.ckpt',
                               num_classes=1001)

        if 'vgg_16' == self.model_name:
            from nets import vgg
            # from tensorflow.contrib.slim.nets import vgg
            return apply_model(x_vgg,
                               vgg.vgg_16,
                               vgg.vgg_arg_scope,
                               'vgg_16.ckpt',
                               no_background=True)

        if 'adv_inception_v3' == self.model_name:
            from nets import inception_v3
            # if __name__ == '__main__':
            # from tensorflow.contrib.slim.nets import inception as inception_v3
            return apply_model(x_inception,
                               inception_v3.inception_v3,
                               inception_v3.inception_v3_arg_scope,
                               'adv_inception_v3.ckpt',
                               num_classes=1001)

        if 'base_inception_model' == self.model_name:
            # from nets import inception_v3
            from tensorflow.contrib.slim.nets import inception as inception_v3
            # import inception_v3
            return apply_model(x_inception,
                               inception_v3.inception_v3,
                               inception_v3.inception_v3_arg_scope,
                               'inception_v3.ckpt',
                               num_classes=1001)
        if 'ens4_adv_inception_v3' == self.model_name:
            from nets import inception_v3
            # from tensorflow.contrib.slim.nets import inception as inception_v3
            return apply_model(x_inception,
                               inception_v3.inception_v3,
                               inception_v3.inception_v3_arg_scope,
                               'ens4_adv_inception_v3.ckpt',
                               num_classes=1001)
        if 'ens_adv_inception_resnet_v2' == self.model_name:
            from nets import inception_resnet_v2
            return apply_model(x_inception,
                               inception_resnet_v2.inception_resnet_v2,
                               inception_resnet_v2.inception_resnet_v2_arg_scope,
                               'ens_adv_inception_resnet_v2.ckpt')

# def load_data():
#     print 'loading dataset',
#     nhwc = np.load('../imagenet/val_5k_images.npy', 'r')
#     labels = np.load('../imagenet/val_5k_labels.npy', 'r')
#     print 'done'
#     return nhwc, labels
import PIL.Image
import StringIO
def load_image(img_path):
    image_name = img_path.split('/')[-1]
    label = image_name.split('_')[-1]
    img = PIL.Image.open(img_path)

    # d_img = StringIO.StringIO()
    # img.save(d_img, "JPEG", quality=90)
    # img = PIL.Image.open(d_img)

    # big_dim = max(img.width, img.height)
    wide = img.width > img.height
    # ratio = 1
    # img = img.resize((int(img.width/ratio), int(img.height/ratio)), resample=1) # PIL.Image.BILINEAR
    print(img.height, img.width)
    new_w = 299 if not wide else int(img.width * 299 / img.height)
    new_h = 299 if wide else int(img.height * 299 / img.width)
    img = img.resize((new_w, new_h), resample=1).crop((0, 0, 299, 299))

    img = np.asarray(img)
    # return img, np.array([int(label)+1]), image_name

    # img = (np.asarray(img) / 255.0).astype(np.float32)
    return img
def npy2png(npy_src_path, save_folder):
    img = np.load(npy_src_path)
    img_name = npy_src_path.split('/')[-1][:-4]
    save_path = save_folder + img_name
    scipy.misc.imsave(save_path+'.png', img)
    return True
def load_image_noise_mul(img_path, factor=1):
    image_name = img_path.split('/')[-1][:-4]
    image_foder = img_path[:-len(image_name)-4]
    label = image_name.split('_')[-1]
    im = PIL.Image.open(img_path)
    im = np.asarray(im)/255.
    npy_path = img_path[:-4]+'.npy'
    im = np.load(npy_path)
    diff_path = image_foder+'diff_'+image_name+'.npy'
    diff = np.load(diff_path)
    clean_img = im - diff[0]
    # diff[0][:60, :60, :] = 0
    mul_img = np.clip(clean_img+diff[0]*factor, 0, 1)
    # mul_img = np.clip(clean_img, 0, 1)
    scipy.misc.imsave(image_foder+'mul_'+str(factor)+'_'+image_name+'.png', mul_img)
    # return im
    return mul_img
def alpha_blend_2_img(img_path1, img_path2, ratio1=0.5, ratio2=0.5):
    im1 = PIL.Image.open(img_path1)
    im2 = PIL.Image.open(img_path2)
    im1 = np.asarray(im1)/255. * ratio1
    im2 = np.asarray(im2)/255. * ratio2
    im = im1 + im2
    # im_name1 = img_path1.split('/')[-1]
    # im_name2 = img_path2.split('/')[-1]
    # img_folder =
    # scipy.misc.imsave()
    return im
# def load_img():
#     img_path = '/home/line290/Documents/project/ImageNet10/chow, chow chow_260'
#     img_path2 = '/home/line290/Documents/project/ImageNet10/matchstick_644'
#     img1 = PIL.Image.open(img_path)
#     img2 = PIL.Image.open(img_path2)
#     img = np.zeros((2,299,299,3))
#     img[0] = np.asarray(img1)
#     img[1] = np.asarray(img2)
#     label = np.array([261, 645])
#     return img, label
if __name__ == '__main__':
    import os
    import PIL.Image
    # from_path = 'dalmatian, coach dog, carriage dog_251_adv_l2_[0.47617173].npy'
    orig_img_path_prefix = '/media/line290/38284FF5284FB0A4/store/dataset/ori_img_1000/'
    src_lst_path = '/media/line290/38284FF5284FB0A4/store/dataset/optens_20_npy.lst'
    save_path = '/media/line290/38284FF5284FB0A4/store/dataset/optens_20_cofds_0_img/'
    f = open(src_lst_path, 'r')
    src_img_paths = f.readlines()
    f.close()
    for img_path in src_img_paths:
        img_path = img_path.split('\n')[0]
        img_name = img_path.split('/')[-1][:-4]
        temp = img_name.split("_")
        orig_name = temp[0]+"_"+temp[1]
        orig_img_path = orig_img_path_prefix + orig_name
        im = PIL.Image.open(orig_img_path)
        im = np.asarray(im)/255.
        img_folder = save_path + img_name + '/'
        if os.path.exists(img_folder) == False:
            os.mkdir(img_folder)
        scipy.misc.imsave(img_folder+orig_name+'.png', im)
        opt_img = np.load(img_path)
        scipy.misc.imsave(img_folder+img_name+'.png', opt_img)
    # save_folder = './'
    # npy2png(from_path, save_folder)