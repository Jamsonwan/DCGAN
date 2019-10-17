from __future__ import division
from __future__ import print_function

import os
import math
import numpy as np
import tensorflow as tf

from glob import glob
from six.moves import xrange
from utils import batch_norm, linear, conv2d_transpose, lrelu, conv2d, get_image, save_images


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
    def __init__(self, sess, image_size=64, is_crop=False, batch_size=64, sample_size=64, lowres=8, z_dim=100,
                 gf_dim=64, df_dim=64, gfc_dim=1024, dfc_dim=1024, c_dim=3, checkpoint_dir=None, lam=0.1):
        """

        :param sess: TensorFlow session
        :param batch_size: the size of batch. Should be specified before training
        :param lowres: (optional) Low resolution image/mask shrink factor.
        :param gf_dim: (optional)Dimension of generator filters in first conv layer
        :param df_dim: (optional)Dimension of discriminator filters in first conv layer
        :param gfc_dim: (optional)Dimension of gen units for fully connected layer
        :param dfc_dim: (optional)Dimension of discrim units for fully connected layer
        :param c_dim: (optional) Dimension of image color. For grayscale input, set to 1.
        """
        self.sess = sess
        self.is_crop = is_crop

        self.batch_size = batch_size
        self.image_size = image_size
        self.sample_size = sample_size
        self.image_shape = [image_size, image_size, c_dim]

        self.lowres = lowres
        self.lowres_size = image_size // lowres
        self.lowres_shape = [self.lowres_size, self.lowres_size, c_dim]

        self.z_dim = z_dim

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.gfc_dim = gfc_dim
        self.dfc = dfc_dim

        self.lam = lam
        self.c_dim = c_dim

        self.d_bns = [batch_norm(name='d_bn{}'.format(format(i, )) for i in range(4))]

        log_size = int(math.log(image_size) / math.log(2))

        self.g_bns = [batch_norm(name='g_bn{}'.format(format(i, )) for i in range(log_size))]

        self.checkpoint_dir = checkpoint_dir
        self.build_model()
        self.model_name = 'DCGAN.model'

    def build_model(self):
        pass

    def train(self, config):
        pass

    def complete(self, config):
        def make_dir(name):
            p = os.path.join(config.outDir, name)
            if not os.path.exists(p):
                os.makedirs(p)

        make_dir('hats_imgs')
        make_dir('completed')
        make_dir('logs')

        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()
        isLoaded = self.load(self.checkpoint_dir)
        assert isLoaded

        nImgs = len(config.imgs)

        batch_idxs = int(np.ceil(nImgs / self.batch_size))
        lowres_mask = np.zeros(self.lowres_shape)

        if config.maskType == 'random':
            fraction_masked = 0.2
            mask = np.ones(self.image_shape)
            mask[np.random.random(self.image_shape[:2]) < fraction_masked] = 0.0
        elif config.maskType == 'center':
            assert (config.centerScale <= 0.5)
            mask = np.ones(self.image_shape)
            sz = self.image_size
            l = int(self.image_size * config.centerScale)
            u = int(self.image_size * (1 - config.centerScale))
            mask[l:u, l:u, :] = 0.0
        elif config.maskType == 'left':
            mask = np.ones(self.image_shape)
            c = self.image_size // 2
            mask[:, :c, :] = 0.0
        elif config.maskType == 'full':
            mask = np.ones(self.image_shape)
        elif config.maskType == 'grid':
            mask = np.zeros(self.image_shape)
            mask[::4, ::4, :] = 1.0
        elif config.maskType == 'lowres':
            lowres_mask = np.ones(self.lowres_shape)
            mask = np.zeros(self.image_shape)
        else:
            assert False

        for idx in xrange(0, batch_idxs):
            l = idx * self.batch_size
            u = min((idx + 1) * self.batch_size, nImgs)
            batchSz = u -1

            batch_files = config.imgs[l: u]
            batch = [get_image(batch_file, self.image_size, is_crop=self.is_crop) for batch_file in batch_files]

            batch_images = np.array(batch).astype(tf.float32)

            if batchSz < self.batch_size:
                print(batchSz)
                padSz = ((0, int(self.batch_size - batchSz)), (0, 0), (0, 0), (0, 0))
                batch_images = np.pad(batch_images, padSz, 'constant')
                batch_images = batch_images.astype(tf.float32)

            z_hats = np.random.uniform(-1, 1, size=(self.batch_size, self.z_dim))
            m = 0
            v = 0

            nRows = np.ceil(batchSz / 8)
            nCols = min(8, batchSz)

            save_images(batch_images[:batchSz, :, :, :], [nRows, nCols], os.path.join(config.outDir, 'before.png'))
            masked_images = np.multiply(batch_images, mask)
            save_images(masked_images[:batchSz, :, :, :], [nRows, nCols], os.path.join(config.outDir, 'masked_png'))

            if lowres_mask.any():
                lowres_images = np.reshape(batch_images, [self.batch_sizem, self.lowres_size, self.lowres,
                                                          self.lowres_size, self.lowres, self.c_dim]).mean(4).mean(2)
                lowres_images = np.multiply(lowres_images, lowres_mask)
                lowres_images = np.repeat(np.repeat(lowres_images, self.lowres, 1), self.lowres, 2)

                save_images(lowres_images[:batchSz, :, :, :], [nRows, nCols], os.path.join(config.outDir, 'lowres.png'))
            for img in range(batchSz):
                with open(os.path.join(config.outDir, 'logs/hats_{:02d}.log'.format(img)), 'a') as f:
                    f.write('iter loss '+ " ".join(['z{}'.format(zi) for zi in range(self.z_dim)]) + '\n')
            for i in xrange(config.nIter):
                fd = {
                    self.z: zhats,
                    self.mask: mask,
                    self.lowres_mask: lowres_mask,
                    self.images: batch_images,
                    self.is_training: False
                }
                run = [self.complete_loss, self.grad_complete_loss, self.G, self.lowres_G]
                loss, g, G_imgs, lowres_G_imgs = self.sess.run(run, feed_dict=fd)

                for img in range(batchSz):
                    with open(os.path.join(config.outDir, 'logs/hats_{:02d}.log'.format(img)), 'ab') as f:
                        f.write('{} {} '.format(i, loss[img]).encode())
                        np.savetxt(f, zhats[img:img + 1])

                if i % config.outInterval == 0:
                    print(i, np.mean(loss[0:batchSz]))
                    imgName = os.path.join(config.outDir,
                                           'hats_imgs/{:04d}.png'.format(i))
                    nRows = np.ceil(batchSz / 8)
                    nCols = min(8, batchSz)
                    save_images(G_imgs[:batchSz, :, :, :], [nRows, nCols], imgName)
                    if lowres_mask.any():
                        imgName = imgName[:-4] + '.lowres.png'
                        save_images(np.repeat(np.repeat(lowres_G_imgs[:batchSz, :, :, :],
                                                        self.lowres, 1), self.lowres, 2),
                                    [nRows, nCols], imgName)

                    inv_masked_hat_images = np.multiply(G_imgs, 1.0 - mask)
                    completed = masked_images + inv_masked_hat_images
                    imgName = os.path.join(config.outDir,
                                           'completed/{:04d}.png'.format(i))
                    save_images(completed[:batchSz, :, :, :], [nRows, nCols], imgName)

                if config.approach == 'adam':
                    # Optimize single completion with Adam
                    m_prev = np.copy(m)
                    v_prev = np.copy(v)
                    m = config.beta1 * m_prev + (1 - config.beta1) * g[0]
                    v = config.beta2 * v_prev + (1 - config.beta2) * np.multiply(g[0], g[0])
                    m_hat = m / (1 - config.beta1 ** (i + 1))
                    v_hat = v / (1 - config.beta2 ** (i + 1))
                    zhats += - np.true_divide(config.lr * m_hat, (np.sqrt(v_hat) + config.eps))
                    zhats = np.clip(zhats, -1, 1)

                elif config.approach == 'hmc':
                    # Sample example completions with HMC (not in paper)
                    zhats_old = np.copy(zhats)
                    loss_old = np.copy(loss)
                    v = np.random.randn(self.batch_size, self.z_dim)
                    v_old = np.copy(v)

                    for steps in range(config.hmcL):
                        v -= config.hmcEps / 2 * config.hmcBeta * g[0]
                        zhats += config.hmcEps * v
                        np.copyto(zhats, np.clip(zhats, -1, 1))
                        loss, g, _, _ = self.sess.run(run, feed_dict=fd)
                        v -= config.hmcEps / 2 * config.hmcBeta * g[0]

                    for img in range(batchSz):
                        logprob_old = config.hmcBeta * loss_old[img] + np.sum(v_old[img] ** 2) / 2
                        logprob = config.hmcBeta * loss[img] + np.sum(v[img] ** 2) / 2
                        accept = np.exp(logprob_old - logprob)
                        if accept < 1 and np.random.uniform() > accept:
                            np.copyto(zhats[img], zhats_old[img])

                    config.hmcBeta *= config.hmcAnneal

                else:
                    assert (False)

    def discriminator(self, image, reuse=False):
        with tf.variable_scope('discriminator') as scope:
            if reuse:
                scope.reuse_variables()

            h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
            h1 = lrelu(self.d_bns[0](conv2d(h0, self.df_dim * 2, name='d_h1_conv'), self.is_training))
            h2 = lrelu(self.d_bns[1](conv2d(h1, self.df_dim * 4, name='d_h2_conv'), self.is_training))
            h3 = lrelu(self.d_bns[2](conv2d(h2, self.df_dim * 8, name='d_h3_conv'), self.is_training))
            h4 = linear(tf.reshape(h3, [-1, 8192]), 1, 'd_h4_lin')

            return tf.nn.sigmoid(h4), h4

    def generator(self, z):
        """
        :param z: the noise vector
        :return:
        """
        with tf.variable_scope("generator") as scope:
            self.z_, self.h0_w, self.h0_b = linear(z, self.gf_dim * 8 * 4 * 4, 'g_h0_lin', with_w=True)
            hs = [None]
            hs[0] = tf.reshape(self.z_, [-1, 4, 4, self.gf_dim * 8])
            hs[0] = tf.nn.relu(self.g_bns[0](hs[0], self.is_training))

            i = 1
            depth_mul = 8
            size = 8

            while size < self.image_size:
                hs.append(None)
                name = 'g_h{}'.format(i)
                hs[i], _, _, conv2d_transpose(hs[i - 1], [self.batch_size, size, size, self.gf_dim * depth_mul],
                                              name=name, with_w=True)
                hs[i] = tf.nn.relu(self.g_bns[i](hs[i]), self.istraining)
                i += 1
                depth_mul //= 2
                size *= 2

            hs.append(None)
            name = 'g_h{}'.format(i)
            hs[i], _, _ = conv2d_transpose(hs[i - 1], [self.batch_size, size, size, 3], name=name, with_w=True)

            return tf.nn.tanh(hs[i])

    def save(self, checkpoint_dir, step):
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_name), global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)

        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            return True
        else:
            return False
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
