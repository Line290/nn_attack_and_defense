import numpy as np
import tensorflow as tf
import tqdm
import PIL.Image
import scipy.misc
imin = 0.
imax = 1.
ih = 299
iw = 299
ic = 3
ifeat = ih * iw * ic
num_classes = 1001

# cg_num_per = 100
# cg_tau = 0.02
cg_num_per = 1
# batchsize = 20
batchsize = 1
# cg_tau = 0.02
cg_tau = 0.0

# usage: python eval_cg.py <model> <dataset name> <images path> <labels path> <count>
import sys
modelname, dataset, images_path, labels_path, orig_img_path, adv_img_path, count = sys.argv[1:]
modelname = 'base_inception_model'
# modelname = 'resnet_v2_50'
# modelname = 'inception_resnet_v2'
# modelname = 'adv_inception_v3'
count = int(count)

import stuff_v2 as stuff
sess = stuff.get_session()
m = stuff.M(modelname)

x = tf.placeholder(shape=[ifeat], dtype=tf.float32)
perturbation = tf.random_uniform((batchsize, ifeat), minval=-cg_tau, maxval=cg_tau, dtype=tf.float32)
avg_rms = tf.reduce_mean(tf.sqrt(tf.reduce_mean(tf.square(perturbation), axis=1)))
x_samples = tf.clip_by_value(x[None] + perturbation, imin, imax)


input_tensor = tf.reshape(x_samples, [batchsize, ih, iw, ic])

logits, end_points = m.predict(tf.reshape(x_samples, [batchsize, ih, iw, ic]))

#logits = tf.nn.softmax(logits,dim=1)
# preds = tf.argmax(logits, axis=1)
# pred_count = tf.unsorted_segment_sum(data=tf.ones(cg_num_per, dtype=tf.int32), segment_ids=preds, num_segments=num_classes)
if images_path[-4:] != '.npy':
    from stuff_v2 import load_image, load_image_noise_mul
    xs = load_image(images_path)
    xs = xs/255.
    # adv_img = load_image(labels_path)
    # adv_img = adv_img / 255.
    #
    # orig_img = load_image(orig_img_path)/255.
    # adv_img2 = load_image(adv_img_path) / 255.
    # xs = orig_img + (xs - orig_img)*0 + (adv_img - orig_img) + (adv_img2 - orig_img)
    # xs = load_image(images_path)/255.
    # xs = adv_img2
    # xs = orig_img
    # alpha = 1
    # xs = adv_img * alpha + xs * (1-alpha)
    # xs = adv_img/255. - xs
    # xs = (xs - xs.min()) / (xs.max() - xs.min())
    # scipy.misc.imsave('230_adv_noise.png', xs)
    # xs = load_image_noise_mul(images_path, factor=1)
    print 'yes'
else:
    xs = np.load(images_path, 'r')
    # scipy.misc.imsave('65_2_907.png', xs)
    # img = PIL.Image.open('65_2_907.png')
    # xs = xs * 255
    # print xs[:10,0,0]
    # print np.asarray(img)[:10,0,0]
    # print np.sum(np.asarray(img) - xs)/ 299/299/3
    # xs = xs / 255.
    # xs = np.asarray(img)/255.
if xs.shape[1] != ifeat:
    xs = xs.reshape((-1, ifeat))
# ys = np.load(labels_path, 'r')
# ys = np.array(int(labels_path))

# import os
# if not os.path.isdir('cgrc_%s' % modelname):
#     os.mkdir('cgrc_%s' % modelname)

pred_counts_all = np.zeros((count, num_classes), dtype=np.int32)
top_preds = np.zeros(count, dtype=np.uint16)
rms_all = np.zeros(count, dtype=np.float32)
for j in tqdm.trange(count, leave=False):
    # pred, pred_counts_all[j], rms_all[j] = sess.run([preds, pred_count, avg_rms], feed_dict={x: xs[j]})
    # top_preds[j] = np.argmax(pred_counts_all[j])
    for i in range(int(cg_num_per/batchsize)):
        temp_pred_logits, all_outs = sess.run([logits, end_points], feed_dict={x:xs[j]})
        if i == 0:
            pred_logits = temp_pred_logits
        else:
            pred_logits = np.vstack((pred_logits, temp_pred_logits))
    # for i, one_out_key in enumerate(end_points_key):
    #     print 'No. ', i, 'var is ', all_outs[one_out_key].shape

    preds = pred_logits.argmax(axis=1)
    top_logits_idx = pred_logits.argsort(axis=1)[:,::-1][:,:20]
    top_logits = pred_logits[range(preds.shape[0]),top_logits_idx]
    preds = np.append(preds, 1000)
    pred_counts_all[j] = np.bincount(preds)
    pred_counts_all[j][1000] -= 1
    top_preds[j] = np.argmax(pred_counts_all[j])
# print type(pred_logits)
print top_preds
print preds
print top_logits_idx
print top_logits
classbook = eval(open('./classes.txt').read())
for i in range(top_logits_idx.shape[1]):
    print 'logit '+str(top_logits[0,i])[:4] +' pred label: ', top_logits_idx[0,i] - 1, ' pred name: ', classbook[top_logits_idx[0,i]-1]
# np.save('cgrc_%s/%s_count.npy' % (modelname, dataset), pred_counts_all)
# print 'accuracy', np.count_nonzero(np.equal(top_preds, ys[:count])) / float(count), 'avg rms', np.mean(rms_all)
