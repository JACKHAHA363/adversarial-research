import keras
import keras.backend as K
import tensorflow as tf
import utils
import logging
import argparse
import time
import math
from utils import str2bool
from utils import save_vae
import numpy as np
import os
# dealing with logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.info("creating log")

# argument
p = argparse.ArgumentParser("a mnist train script")
p.add_argument('--data-dir', default='/mnt/data-1/yuchen.lu/mnist_gzip/')
p.add_argument('--batch-size', default=1000, type=int)
p.add_argument('--lr', default=0.001, type=float)
p.add_argument('--num-epoch', default=1000, type=int)
p.add_argument('--model-prefix', default='./vae_models/vae', type=str)
p.add_argument('--verbose', default='true', type=str)
args = p.parse_args()

# get data
data_train, label_train, data_val, label_val = utils.get_data(args.data_dir)

# prepare sess
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)

# build model
from keras.layers import Dense, Flatten, Input, Lambda
from keras.models import Model, Sequential
def vae():
    num_hidden = 50
    x = Input(shape=(28,28,1))
    # encoder
    h = Dense(256, activation='relu')(Flatten()(x))
    z_mean = Dense(num_hidden)(h)
    z_log_var = Dense(num_hidden)(h)
    # sampling
    def sampling(args):
        (mean, log_var) = args
        eps = K.random_normal(shape=K.shape(mean))
        tmp = mean + K.exp(log_var / 2) * eps
        return tmp
    z = Lambda(sampling)([z_mean, z_log_var])
    # decoder layers
    decoder_h = Dense(256, activation='relu', input_shape=(num_hidden,))
    decoder_h2 = Dense(256, activation='relu')
    decoder_mean = Dense(784, activation='sigmoid')
    # decoder computation
    h_decoded = decoder_h(z)
    h_decoded = decoder_h2(h_decoded)
    x_decoded_mean = decoder_mean(h_decoded)
    full = Model(inputs=x, outputs=x_decoded_mean)
    encoder = Model(inputs=x, outputs=[z_mean, z_log_var])

    # reuse previous layer to get a decoder
    decoder = Sequential()
    decoder.add(decoder_h)
    decoder.add(decoder_h2)
    decoder.add(decoder_mean)
    return full, encoder, decoder

model, enc, dec = vae()
data = tf.placeholder(tf.float32, shape=(None, 28, 28, 1), name='data')
reconst = model(data)
mean, log_var = enc(data)
# define loss
data_reshape = tf.reshape(data, [-1, 784])
reconst_loss = -tf.reduce_mean(
    tf.reduce_sum(data_reshape * tf.log(reconst + 1e-10) +
                  (1-data_reshape)*tf.log(1-reconst + 1e-10), axis=1)
)
kl_loss = - 0.5 * tf.reduce_mean(tf.reduce_sum(
    (1 + log_var - tf.square(mean) - tf.exp(log_var)),
    axis=1
))
loss = reconst_loss + kl_loss

# optimizer
opt = tf.train.RMSPropOptimizer(args.lr)
train_step = opt.minimize(loss)

from cleverhans import utils_tf
utils_tf.initialize_uninitialized_global_variables(K.get_session())
# Compute number of batches
nb_batches = int(math.ceil(float(len(data_train)) / args.batch_size))
step = 0
lowest_loss = np.inf
for epoch in range(args.num_epoch):
    # training
    if str2bool(args.verbose):
        logging.info("Epoch {} start training...".format(epoch))

    prev = time.time()
    indices = np.random.choice(len(data_train), len(data_train), False)
    for batch in range(nb_batches):
        # Compute batch start and end indices
        start, end = utils_tf.batch_indices(
            batch, len(data_train), args.batch_size)
        # prepare feed dict
        feed_dict = {
            data: data_train[indices[start:end]],
            K.learning_phase(): 1,
        }
        run_result = sess.run(
            [loss, train_step, reconst_loss],
            feed_dict=feed_dict,
        )
        loss_eval = run_result[0]
        step += 1
    cur = time.time()
    val_loss_eval = sess.run(loss, {data:data_val, K.learning_phase():0})
    if str2bool(args.verbose):
        logging.info("Epoch take {} seconds, train LB {}, val LB {}".format(cur-prev, -loss_eval, -val_loss_eval))
    prev = cur
    if val_loss_eval < lowest_loss:
        lowest_loss = val_loss_eval
        if args.model_prefix is not None:
            save_vae(enc, dec, args.model_prefix)
            logging.info("checkpointed")

# visualize shit
imgs = data_val[6:6+5]
reconst_imgs = sess.run(tf.reshape(reconst,(-1,28,28,1)), {data:imgs,
                                                 K.learning_phase():0}
                       )
canvas = np.zeros([28*5, 28*2])
canvas[:, 0:28] = np.concatenate(imgs).reshape([140, 28])
canvas[:, 28:56] = np.concatenate(reconst_imgs).reshape([140, 28])

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.imshow(canvas, cmap='gray')
plt.savefig("vae.png")
