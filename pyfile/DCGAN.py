from __future__ import division
from __future__ import print_function

import os
import math
import numpy as np
import tensorflow as tf

from glob import glob
from six.moves import xrange


def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))


def gen_random(model, size):
    if model == 'normal01':
        return np.random.normal(0, 1, size=size)
    if model == 'uniform_signed':
        return np.random.uniform(-1, 1, size=size)
    if model == 'uniform_unsigned':
        return np.random.uniform(0, 1, size=size)


class DCGAN(object):
    def __init__(self, sess, input_height=108, input_width=108, crop=True, batch_size=64, sample_num=64,
                 output_height=64, output_width=64, y_dim=None, z_dim=100, gf_dim=64, df_dim=64, gfc_dim=1024,
                 dfc_dim=1024, c_dim=3, dataset_name='default', max_to_keep=1, input_fname_pattern='*.jpg',
                 checkpoint_dir='ckpts', sample_dir='samples', out_dir='./out', data_dir='./data'):
        """

        :param sess: TensorFlow session
        :param batch_size: the size of batch. Should be specified before training
        :param gf_dim: (optional)Dimension of generator filters in first conv layer
        :param df_dim: (optional)Dimension of discriminator filters in first conv layer
        :param gfc_dim: (optional)Dimension of gen units for fully connected layer
        :param dfc_dim: (optional)Dimension of discrim units for fully connected layer
        :param c_dim: (optional) Dimension of image color. For grayscale input, set to 1.
        """
        self.sess = sess
        self.crop = crop

        self.batch_size = batch_size
        self.sample_num = sample_num

        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width

        self.y_dim = y_dim
        self.z_dim = z_dim

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.gfc_dim = gfc_dim
        self.dfc = dfc_dim





# def transpose_con2d(x, output_space):
#     return tf.layers.conv2d_transpose(x, output_space, kernel_size=5, strides=2, padding='same',
#                                       kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))
#
#
# def generator(z, output_dim, reuse=False, training=True):
#     """
#     the structure of generator network
#     :param z: random vector z
#     :param output_dim: output dimension of generator
#     :param reuse: whether use the model variables of existing model
#     :param training: controlling the batch normalization
#     :return: like generated graph
#     """
#
#     with tf.variable_scope('generator', reuse=reuse):
#         fc1 = tf.layers.dense(z, 4*4*512)
#         fc1 = tf.reshape(fc1, (-1, 4, 4, 512))
#         fc1 = tf.layers.batch_normalization(fc1, training=training)
#         fc1 = tf.nn.relu(fc1)
#         # fc1[fc1 == 0] = alpha
#
#         t_conv1 = transpose_con2d(fc1, 256)
#         t_conv1 = tf.layers.batch_normalization(t_conv1, training=training)
#         t_conv1 = tf.nn.relu(t_conv1)
#         # t_conv1[t_conv1 == 0] = alpha
#
#         t_conv2 = transpose_con2d(t_conv1, 128)
#         t_conv2 = tf.layers.batch_normalization(t_conv2, training=training)
#         t_conv2 = tf.nn.relu(t_conv2)
#
#         logits = transpose_con2d(t_conv2, output_dim)
#
#         out = tf.tanh(logits)
#
#         return out
#
#
# def lrelu(x, alpha=0.2):
#     return tf.maximum(alpha*x, x)
#
#
# def discriminator(x, reuse=False, alpha=0.2, training=True):
#     """
#     the discriminator of network
#     :param x: the output of generator and the true data
#     :param reuse:  whether use the model variables of existing model
#     :param alpha: scalar of lrelu
#     :param training: controlling the batch normalization
#     :return: sigmoid probabilities and logits
#     """
#     with tf.variable_scope('discriminator', reuse=reuse):
#         conv1 = tf.nn.conv2d(x, 64)
#         conv1 = lrelu(conv1, alpha)
#
#         conv2 = tf.nn.conv2d(conv1, 128)
#         conv2 = tf.layers.batch_normalization(conv2, training=training)
#         conv2 = lrelu(conv2, alpha)
#
#         conv3 = tf.nn.conv2d(conv2, 256)
#         conv3 = tf.layers.batch_normalization(conv3, training=training)
#         conv3 = lrelu(conv3, alpha)
#
#         flat = tf.reshape(conv3, (-1, 4**4*256))
#         logits = tf.layers.dense(flat, 1)
#
#         out = tf.sigmoid(logits)
#
#         return out, logits
#
#
# def model_loss(input_real, input_z, output_dim, alpha=0.2, smooth=0.1):
#     """
#     calculate the loss of discriminator and generator
#     :param input_real: images from real dataset
#     :param input_z: random vector z
#     :param output_dim: channels of output images
#     :param alpha: lerelu scalar
#     :param smooth: label smoothing scalar
#     :return:discriminator loss, generator loss
#     """
#     g_output = generator(input_z, output_dim)
#
#     d_output_real, d_logits_real = discriminator(input_real, alpha=alpha)
#
#     d_output_fake, d_logits_fake = discriminator(g_output, reuse=True, alpha=alpha)
#
#     d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real,
#                                                                          labels=tf.ones_like(d_logits_real)*(1 - smooth)))
#     # notice
#     d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,
#                                                                          labels=tf.zeros_like(d_output_fake)))
#
#     g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,
#                                                                     labels=tf.ones_like(d_output_fake)))
#     d_loss = d_loss_fake + d_loss_real
#
#     return d_loss, g_loss
#
#
# if __name__ == '__main__':
#     pass