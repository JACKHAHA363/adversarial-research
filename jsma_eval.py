from evaluate import jsma_eval
from evaluate import logits_stats
import tensorflow as tf
import numpy as np
import keras.backend as K
import argparse
import utils
from metrics import Accuracy
import cleverhans.attacks_tf
import os
from utils import str2bool

import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

p = argparse.ArgumentParser("evaluation of jsma")
p.add_argument('--models', nargs='*', type=str)
p.add_argument('--steps', default=1, type=int)
p.add_argument('--num-samples', default=10, type=int)
p.add_argument('--theta', default=8, type=float)
p.add_argument('--gamma', default=0.143, type=float)
p.add_argument('--data-dir',
               default='/mnt/data-1/yuchen.lu/mnist_gzip/')
p.add_argument('--save-dir', default=None, type=str)
args = p.parse_args()

#############################################
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

_, _, data_val, label_val = utils.get_data(args.data_dir)

for m in args.models:
    model = utils.load_model(m)
    model_name = os.path.basename(m)
    data = tf.placeholder(tf.float32, shape=(None, 28, 28, 1), name='data')
    label = tf.placeholder(tf.float32, shape=(None, 10), name='label')
    #jsma_eval(
    #    data=data,
    #    logits=model(data) / 100,
    #    sess=K.get_session(),
    #    data_val=data_val,
    #    label_val=label_val,
    #    args=args,
    #)

    logits_stats(
        data=data, logits=model(data), sess=K.get_session(),
        data_val=data_val,
    )
