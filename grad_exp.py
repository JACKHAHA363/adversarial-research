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

######### parse arg #########################
p = argparse.ArgumentParser("check gradient")
p.add_argument('--sources', nargs='*', type=str)
p.add_argument('--eps', default=0.3, type=float)
p.add_argument('--steps', default=1, type=int)
p.add_argument('--batch-size', default=100, type=int)
p.add_argument('--num-samples', default=10000, type=int)
p.add_argument('--data-dir',
               default='/mnt/data-1/yuchen.lu/mnist_gzip/')
p.add_argument('--save-dir', default=None, type=str)
p.add_argument('--rand', default='false', type=str)
args = p.parse_args()

#############################################
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

############### build attack graph ############
def fgm(model, x, y, steps, eps, clip_min, clip_max, batch_size):
    ''' use true label '''
    adv_x = x

    if str2bool(args.rand) is True:
        jump = tf.random_normal([batch_size, 28, 28, 1])
        jump = 0.05 * tf.sign(jump)
        adv_x = adv_x + jump
        eps = eps - 0.05

    step_size = eps / steps
    for _ in range(steps):
        logits = model(adv_x)
        logits = logits / 100
        loss = tf.losses.softmax_cross_entropy(
            onehot_labels = y,
            logits = logits,
        )
        grad,  = tf.gradients(loss, adv_x)
        scaled_signed_grad = step_size * tf.sign(grad)
        adv_x = tf.stop_gradient(adv_x + scaled_signed_grad)
        adv_x = tf.clip_by_value(adv_x, clip_min, clip_max)
    return adv_x



_, _, data_val, label_val = utils.get_data(args.data_dir)
indices = np.random.choice(len(data_val), args.num_samples,False)
rand_data_val = data_val[indices]
rand_label_val = label_val[indices]

for src_ind in range(len(args.sources)):
    src = args.sources[src_ind]
    src_model = utils.load_model(src)
    src_name = os.path.basename(src)
    data = tf.placeholder(tf.float32, shape=(None, 28, 28, 1), name='data')
    label = tf.placeholder(tf.float32, shape=(None, 10), name='label')
    logits = src_model(data)
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels = label,
        logits = logits,
    )
    grad, = tf.gradients(loss, data)
    signed_grad = tf.sign(grad)

    grad_eval, signed_grad_eval = K.get_session().run([grad, signed_grad],
                                                      {data:data_val,
                                                       label:label_val,K.learning_phase():0})
    logging.info(
        "percentage of non-zero element {}%".format(float(np.count_nonzero(grad_eval)) /
                                                    np.product(list(grad_eval.shape)) *
                            100))

