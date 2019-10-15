import math
import numpy as np
import json
import random
import pprint
import imageio
import scipy.misc
import tensorflow as tf
import moviepy.editor as mpy

from six.moves import xrange
from time import gmtime, strftime
from tensorflow.python.framework import ops


class batch_norm(object):
    def __init__(self, epsilon=1e-5, momentum=0.9, name='batch_norm'):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, train=True, *args, **kwargs):
        return tf.layers.batch_normalization(x, momentum=self.momentum, epsilon=self.epsilon, scale=True,
                                             training=train, name=self.name)


def binary_cross_entropy(preds, targets, name=None):
    """
    compute binary cross entropy given predictions
    :param preds: A 'Tensor' of type 'float32' or 'float64'
    :param targets: A 'Tensor' of the same type and shape as 'preds'
    :param name:
    :return:
    """
    epsilon = 1e-12
    with ops.op_scope([preds, targets], name, 'bce_loss') as scp:
        preds = ops.convert_to_tensor(preds, name='preds')
        targets = ops.convert_to_tensor(targets, name='targets')
        return tf.reduce_mean(-(targets * tf.log(preds+epsilon) + (1. - targets)*tf.log(1. - preds + epsilon)))


def conv_cond_concat(x, y):
    """
    Concatenate conditioning vector on feature map axis
    :param x:
    :param y:
    :return:
    """
    x_shape = x.get_shape()
    y_shape = y.get_shape()
    return tf.concat(3, [x, y*tf.ones[x_shape[0], x_shape[1], x_shape[2], y_shape[3]]])


def conv2d(input_, output_dim, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, name='conv2d'):
    """

    :param input_:
    :param output_dim:
    :param k_h: conv kernel height
    :param k_w: conv kernel width
    :param d_h: filter's height
    :param d_w: fliter's width
    :param stddev:
    :param name:
    :return:
    """
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

        bias = tf.get_variable('bias', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.bias_add(conv, bias)

        return conv


def conv2d_transpose(input_, output_shape, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, name='conv2d_transpose', with_w=False):
    w = tf.get_variable('W', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                        initializer=tf.random_normal_initializer(stddev=stddev))
    deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape, strides=[1, d_h, d_w, 1])
    bias = tf.get_variable('bias', [output_shape[-1]], initializer=tf.constant_initializer(0.0))

    deconv = tf.nn.bias_add(deconv, bias)

    if with_w:
        return deconv, w, bias
    else:
        return deconv


def lrelu(x, leak=0.2, name='lrelu'):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1*x + f2*abs(x)


def linear(input_, output_size, scope="Linear", stddev=0.02, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope):
        matrix = tf.get_variable('matrix', [shape[1], output_size], tf.float32, tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable('bias', [output_size], initializer=tf.constant_initializer(bias_start))

        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias


def imread(path):
    """
    read a picture as type of RGB
    :param path:
    :return:
    """
    return imageio.imread(path, 'RGB').astype(np.float)


def center_crop(image, image_size, crop_w=None, resize_w=64):
    if crop_w is None:
        crop_w = image_size

    h, w = image.shape[:2]
    j = int(round((h - image_size) / 2.))
    i = int(round((w - crop_w) / 2.))

    return scipy.misc.imresize(image[j:j+image_size, i:i+crop_w], [resize_w, resize_w])


def transform(image, image_size=64, is_crop=True):
    if is_crop:
        cropped_image = center_crop(image, image_size)
    else:
        cropped_image = image

    return np.array(cropped_image) / 127.5 - 1


def get_image(image_path, image_size, is_crop=True):
    return transform(imread(image_path), image_size, is_crop)


def inverse_transform(images):
    return (images + 1.) / 2.


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((int(h * size[0]), int(w * size[1]), 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h. i*w:i*w+w, :] = image
    return img


def save_images(images, size, image_path):
    img = merge(inverse_transform(images), size)
    return imageio.imsave(image_path, (255*img).astype(np.uint8))


def to_json(output_path, *layers):
    with open(output_path, "w") as layer_f:
        lines = ""
        for w, b, bn in layers:
            layer_idx = w.name.split('/')[0].split('h')[1]
            B = b.eval()

            if "lin/" in w.name:
                W = w.eval()
                depth = W.shape[1]
            else:
                W = np.rollaxis(w.eval(), 2, 0)
                depth = W.shape[0]
            bias = {'sy': 1, "sx": 1, "depth": depth, 'w': ['%.2f' % elem for elem in list(B)]}

            if bn is not None:
                gamma = bn.gamma.eval()
                beta = bn.beta.eval()

                gamma = {'sy': 1, 'sx': 1, "depth": depth, 'w': ['%.2f' % elem for elem in list(gamma)]}
                beta = {'sy': 1, 'sx': 1, "depth": depth, 'w': ['%.2f' % elem for elem in list(beta)]}
            else:
                gamma = {"sy": 1, "sx": 1, "depth": 0, "w": []}
                beta = {"sy": 1, "sx": 1, "depth": 0, "w": []}

            if "lin/" in w.name:
                fs = []
                for w in W.T:
                    fs.append({"sy": 1, "sx": 1, "depth": W.shape[0], "w": ['%.2f' % elem for elem in list(w)]})

                lines += """
                                    var layer_%s = {
                                        "layer_type": "fc",
                                        "sy": 1, "sx": 1,
                                        "out_sx": 1, "out_sy": 1,
                                        "stride": 1, "pad": 0,
                                        "out_depth": %s, "in_depth": %s,
                                        "biases": %s,
                                        "gamma": %s,
                                        "beta": %s,
                                        "filters": %s
                                    };""" % (layer_idx.split('_')[0], W.shape[1], W.shape[0], biases, gamma, beta, fs)

            else:
                fs = []
                for w_ in W:
                    fs.append(
                        {"sy": 5, "sx": 5, "depth": W.shape[3], "w": ['%.2f' % elem for elem in list(w_.flatten())]})

                lines += """
                                 var layer_%s = {
                                     "layer_type": "deconv",
                                     "sy": 5, "sx": 5,
                                     "out_sx": %s, "out_sy": %s,
                                     "stride": 2, "pad": 1,
                                     "out_depth": %s, "in_depth": %s,
                                     "biases": %s,
                                     "gamma": %s,
                                     "beta": %s,
                                     "filters": %s
                                 };""" % (layer_idx, 2 ** (int(layer_idx) + 2), 2 ** (int(layer_idx) + 2),
                                          W.shape[0], W.shape[3], bias, gamma, beta, fs)

        layer_f.write(" ".join(lines.replace("'", "").split()))


def make_gif(images, fname, duration=2, true_image=False):
    def make_frame(t):
        try:
            x = images[int(len(images) / duration * t)]
        except:
            x = images[-1]
        if true_image:
            return x.astype(np.uint8)
        else:
            return ((x+1) / 2 * 255).astype(np.unit8)
    clip = mpy.VideoClip(make_frame=make_frame, duration=duration)
    clip.write_gif(fname, fps=len(images) / duration)


def visualize(sess, dcgan, config, option):
    if option == 0:
        z_sample = np.random.uniform(-0.5, 0.5, size=(config.batch_size, dcgan.z_dim))
        samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
        save_images(samples, [8, 8], './samples/test_%s.png' % strftime("%Y-%m-%d %H:%<:%S", gmtime()))
    elif option == 1:
        values = np.arange(0, 1, 1. / config.batch_size)
        for idx in xrange(100):
            print(" [*] %d" % idx)
            z_sample = np.zeros([config.batch_size, dcgan.z_dim])
            for kdx, z in enumerate(z_sample):
                z[idx] = values[kdx]

            samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
            save_images(samples, [8, 8], './samples/test_arange_%s.png' % idx)
    elif option == 2:
        values = np.arange(0, 1, 1. / config.batch_size)
        for idx in [random.randint(0, 99) for _ in xrange(100)]:
            print(" [*] %d" % idx)
            z = np.random.uniform(-0.2, 0.2, size=(dcgan.z_dim))
            z_sample = np.tile(z, (config.batch_size, 1))
            # z_sample = np.zeros([config.batch_size, dcgan.z_dim])
            for kdx, z in enumerate(z_sample):
                z[idx] = values[kdx]

            samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
            make_gif(samples, './samples/test_gif_%s.gif' % idx)
    elif option == 3:
        values = np.arange(0, 1, 1. / config.batch_size)
        for idx in xrange(100):
            print(" [*] %d" % idx)
            z_sample = np.zeros([config.batch_size, dcgan.z_dim])
            for kdx, z in enumerate(z_sample):
                z[idx] = values[kdx]

            samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
            make_gif(samples, './samples/test_gif_%s.gif' % idx)
    elif option == 4:
        image_set = []
        values = np.arange(0, 1, 1. / config.batch_size)

        for idx in xrange(100):
            print(" [*] %d" % idx)
            z_sample = np.zeros([config.batch_size, dcgan.z_dim])
            for kdx, z in enumerate(z_sample): z[idx] = values[kdx]

            image_set.append(sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample}))
            make_gif(image_set[-1], './samples/test_gif_%s.gif' % idx)

        new_image_set = [merge(np.array([images[idx] for images in image_set]), [10, 10]) \
                         for idx in range(64) + range(63, -1, -1)]
        make_gif(new_image_set, './samples/test_gif_merged.gif', duration=8)