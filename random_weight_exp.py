from mnist_graph import get_model
import tensorflow as tf
import numpy as np
import argparse
import cleverhans.utils_tf as utils_tf
import time
import utils
from utils import str2bool
import os
import metrics
import keras.backend as K
import math
import logging
from keras.layers import Dense
# dealing with logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.info("creating log")

# TODO add attacks here https://arxiv.org/pdf/1608.04644.pdf
args = utils.parse_args()
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)

data = tf.placeholder(tf.float32, shape=(None, 28, 28, 1), name='data')
label = tf.placeholder(tf.float32, shape=(None, 10), name='label')
# load student model
model = get_model('conv2_rep')
rep = model(data)
#rep = tf.stop_gradient(rep)
logits = Dense(units=10, activation=None)(rep)

# get data
data_train, label_train, data_val, label_val = utils.get_data(args.data_dir)

from metrics import Accuracy
preds = tf.nn.softmax(logits)
acc = Accuracy(label, preds, args.batch_size)
loss = tf.losses.softmax_cross_entropy(
    onehot_labels = label,
    logits = logits / args.temp,
)
opt = tf.train.AdamOptimizer(learning_rate=args.lr)
train_step = opt.minimize(loss)

# initialize
utils_tf.initialize_uninitialized_global_variables(K.get_session())

# Compute number of batches
nb_batches = int(math.ceil(float(len(data_train)) / args.batch_size))
for epoch in range(args.num_epoch):
    # training
    if str2bool(args.verbose):
        logging.info("Epoch {} start training...".format(epoch))
    # reset metric
    sess.run(acc.reset())

    prev = time.time()
    for batch in range(nb_batches):
        # Compute batch start and end indices
        start, end = utils_tf.batch_indices(
            batch, len(data_train), args.batch_size)
        # prepare feed dict
        feed_dict = {
            data: data_train[start:end],
            label: label_train[start:end],
            K.learning_phase(): 1
        }
        run_list = [loss + acc.update()]
        run_list.append(train_step)
        run_result = sess.run(
            run_list,
            feed_dict=feed_dict,
        )
        loss_eval = run_result[0]
    train_acc = sess.run(acc.get_result())
    # eval on validation
    val_acc = utils.model_eval(
        data=data, label=label,
        data_val=data_val, label_val=label_val,
        metric=acc, args=args,
    )
    cur = time.time()
    if str2bool(args.verbose):
        logging.info("Epoch take {} seconds, train acc {}, val acc {}".format(cur-prev, train_acc * 100, val_acc * 100))
    prev = cur


