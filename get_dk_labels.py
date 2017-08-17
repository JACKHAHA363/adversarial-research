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
######### parse arg #########################
p = argparse.ArgumentParser("calculate soft labels from models")
p.add_argument('--teachers', nargs='*', type=str, default=["./models/A"])
p.add_argument('--batch-size', default=100, type=int)
p.add_argument('--distill-temp', default=1.0, type=float)
p.add_argument('--save-prefix', default=None, type=str)
p.add_argument('--data-dir', default='/mnt/data-1/yuchen.lu/mnist_gzip/')
args = p.parse_args()


# TODO add attacks here https://arxiv.org/pdf/1608.04644.pdf
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)

data = tf.placeholder(tf.float32, shape=(None, 28, 28, 1), name='data')

# build dk label graph
assert args.teachers is not None
teachers = [utils.load_model(teacher_prefix) for teacher_prefix in args.teachers ]
dk_labels = [ tf.nn.softmax(T(data) / args.distill_temp) for T in teachers]
dk_label = tf.add_n(dk_labels) / len(dk_labels)

# get data
data_train, _, data_val, _ = utils.get_data(args.data_dir)

# training
soft_labels_train = utils.model_predict(data, dk_label, data_train, args)
soft_labels_val = utils.model_predict(data, dk_label, data_val, args)

if args.save_prefix is not None:
    import pickle
    dst_dir = os.path.dirname(args.save_prefix)
    if os.path.isdir(dst_dir) is not True:
        os.mkdir(dst_dir)
    filename = args.save_prefix
    with open(filename, 'wb') as f:
        pickle.dump((soft_labels_train, soft_labels_val), f)
