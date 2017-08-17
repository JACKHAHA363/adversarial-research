import tensorflow as tf
import numpy as np
import keras.backend as K
import argparse
import utils
from metrics import Accuracy
import cleverhans.attacks_tf
import os
from utils import str2bool
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import logging
import cleverhans
import scipy.stats

logger = logging.getLogger()
logger.setLevel(logging.INFO)

######### parse arg #########################
p = argparse.ArgumentParser("exp on vae to detect adv samples")
p.add_argument('--vae', default='./vae_models/vae', type=str)
p.add_argument('--num-samples', default=10000, type=int)
p.add_argument('--batch-size', default=100, type=int)
p.add_argument('--data-dir',default='/mnt/data-1/yuchen.lu/mnist_gzip/')
p.add_argument('--adv-data-path', default='./attack_npy/A_temp1.npy')
args = p.parse_args()

#############################################
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

data_train, lebal_train, data_val, label_val = utils.get_data(args.data_dir)
adv_val = np.load(args.adv_data_path)
adv_val = adv_val.reshape([-1, 28,28,1])

indices = np.random.choice(len(data_val), args.num_samples,False)
rand_data_val = data_val[indices]
rand_label_val = label_val[indices]
rand_adv_val = adv_val[indices]

model, enc, dec = utils.load_vae(args.vae)
data = tf.placeholder(tf.float32, shape=(None, 28, 28, 1), name='data')
reconst = model(data)
mean, log_var = enc(data)
# define loss
data_reshape = tf.reshape(data, [-1, 784])
reconst_loss = -tf.reduce_sum(
    data_reshape * tf.log(reconst + 1e-10) + (1-data_reshape)*tf.log(1-reconst + 1e-10),
    axis=1
)
kl_loss = - 0.5 * tf.reduce_sum(
    (1 + log_var - tf.square(mean) - tf.exp(log_var)),
    axis=1
)
loss = reconst_loss + kl_loss
LB = -loss
mean, var = tf.nn.moments(LB, [0])
LB_clean, LB_mean, LB_var = K.get_session().run(
    [LB, mean, var],
    {data: rand_data_val}
)
LB_adv = K.get_session().run(
    [LB],
    {data: rand_adv_val}
)

def p_values(mean, var, samples):
    ''' one-sided p-value of z test '''
    z_scores = (samples - mean) / np.sqrt(var)
    return scipy.stats.norm.sf(abs(z_scores))

p_clean = p_values(LB_mean, LB_var, LB_clean)
p_adv = p_values(LB_mean, LB_var, LB_adv)

confusion = np.zeros([2,2])

alpha = 0.01
p_clean_clean = np.mean(p_clean > alpha)
p_adv_adv = np.mean(p_adv < alpha)

logging.info("clean->clean {}, adv->adv {}".format(p_clean_clean, p_adv_adv))
