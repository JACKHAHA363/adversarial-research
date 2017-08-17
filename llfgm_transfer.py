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
p = argparse.ArgumentParser("transfer evaluation of fgm")
p.add_argument('--sources', nargs='*', type=str)
p.add_argument('--targets', nargs='*', type=str)
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
def target_fgm(model, x, targets, steps, eps, clip_min, clip_max, batch_size):
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
            onehot_labels = targets,
            logits = logits,
        )
        grad,  = tf.gradients(loss, adv_x)
        scaled_signed_grad = step_size * tf.sign(grad)
        adv_x = tf.stop_gradient(adv_x - scaled_signed_grad)
        adv_x = tf.clip_by_value(adv_x, clip_min, clip_max)
    return adv_x


from cleverhans.utils import random_targets
_, _, data_val, label_val = utils.get_data(args.data_dir)
target_val = random_targets(label_val, 10)
indices = np.random.choice(len(data_val), args.num_samples,False)
rand_data_val = data_val[indices]
rand_label_val = label_val[indices]
rand_target_val = target_val[indices]

if args.save_dir is not None:
    if os.path.isdir(args.save_dir) is False:
        os.mkdir(args.save_dir)

for src_ind in range(len(args.sources)):
    src = args.sources[src_ind]
    src_model = utils.load_model(src)
    src_name = os.path.basename(src)
    data = tf.placeholder(tf.float32, shape=(None, 28, 28, 1), name='data')
    label = tf.placeholder(tf.float32, shape=(None, 10), name='label')
    targets = tf.placeholder(tf.float32, shape=(None,10), name='targets')
    batch_size = tf.placeholder(tf.int32, ())
    adv_data = target_fgm(src_model, data, targets, args.steps, args.eps, 0, 1,
                   batch_size)
    src_adv_preds = tf.nn.softmax(src_model(adv_data))
    src_adv_acc = Accuracy(label, src_adv_preds, args.batch_size)
    src_adv_acc_result = utils.model_eval(data, label, rand_data_val,
                                          rand_label_val,src_adv_acc, args,
                                          {batch_size:args.batch_size,targets:
                                          rand_target_val})
    logging.info("{}/{} samples fool {}".format(args.num_samples*(1-src_adv_acc_result) , args.num_samples, src_name))
    new_data_val = rand_data_val
    new_label_val = rand_label_val

    if args.save_dir is not None:
        adv_data_eval = K.get_session().run(
            adv_data,
            {data:data_val,
             label:label_val,
             K.learning_phase():1,
             batch_size:len(data_val)
            }
        )
        np.save(os.path.join(args.save_dir, src_name),
                adv_data_eval.reshape([-1,784]), False)
