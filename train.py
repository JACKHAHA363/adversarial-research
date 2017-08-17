from mnist_graph import get_model
import tensorflow as tf
import numpy as np
import argparse
import cleverhans.utils_tf
import utils
from utils import str2bool
import os
import metrics
import keras.backend as K

import logging
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
model = get_model(args.model)

# get data
data_train, label_train, data_val, label_val = utils.get_data(args.data_dir)
# load dk_labels
soft_label_train = None
soft_label_val = None
if args.dk_label is not None:
    import pickle
    logging.info("load dark knowledge...")
    with open(args.dk_label, 'rb') as f:
        soft_label_train, soft_label_val = pickle.load(f)

# training
utils.model_train(
    data=data, label=label, model=model,
    data_train=data_train, label_train=label_train,
    data_val=data_val, label_val=label_val,
    soft_label_train=soft_label_train,
    soft_label_val=soft_label_val,
    args=args,
)
