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
logger = logging.getLogger()
logger.setLevel(logging.INFO)

######### parse arg #########################
p = argparse.ArgumentParser("exp on vae to detect adv samples")
p.add_argument('--sources', nargs='*', type=str)
p.add_argument('--vae', default='./vae', type=str)
p.add_argument('--eps', default=0.3, type=float)
p.add_argument('--steps', default=1, type=int)
p.add_argument('--batch-size', default=100, type=int)
p.add_argument('--num-samples', default=10000, type=int)
p.add_argument('--data-dir',
               default='/mnt/data-1/yuchen.lu/mnist_gzip/')
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



data_train, lebal_train, data_val, label_val = utils.get_data(args.data_dir)
indices = np.random.choice(len(data_val), args.num_samples,False)
rand_data_val = data_val[indices]
rand_label_val = label_val[indices]

model, enc, dec = utils.load_vae(args.vae)

def VAELoss(model, enc, data):
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
    LB = -loss
    return (LB, reconst_loss, kl_loss)

for src_ind in range(len(args.sources)):
    src = args.sources[src_ind]
    src_model = utils.load_model(src)
    src_name = os.path.basename(src)

    data = tf.placeholder(tf.float32, shape=(None, 28, 28, 1), name='data')
    label = tf.placeholder(tf.float32, shape=(None, 10), name='label')
    batch_size = tf.placeholder(tf.int32, ())
    adv_data = fgm(src_model, data, label, args.steps, args.eps, 0, 1,
                   batch_size)
    clean_vae_loss = VAELoss(model, enc, data)
    clean_vae_eval = K.get_session().run(
        clean_vae_loss,
        {data:rand_data_val, label:rand_label_val, batch_size:args.num_samples, K.learning_phase():0}
    )
    logging.info("{} clean has LB {}, reconst loss {}, kl loss {}".format(
        src_name,
        clean_vae_eval[0],
        clean_vae_eval[1],
        clean_vae_eval[2]
    ))
    clean_reconst = model(data)
    clean_reconst_eval = K.get_session().run(
        clean_reconst,
        {data:rand_data_val, label:rand_label_val, batch_size:args.num_samples, K.learning_phase():0}
    )
    adv_vae_loss = VAELoss(model, enc, adv_data)
    adv_vae_eval = K.get_session().run(
        adv_vae_loss,
        {data:rand_data_val, label:rand_label_val, batch_size:args.num_samples, K.learning_phase():0}
    )
    logging.info("{} adv has LB {}, reconst loss {}, kl loss {}".format(
        src_name,
        adv_vae_eval[0],
        adv_vae_eval[1],
        adv_vae_eval[2]
    ))
    adv_reconst = model(adv_data)
    adv_reconst_eval = K.get_session().run(
        adv_reconst,
        {data:rand_data_val, label:rand_label_val, batch_size:args.num_samples, K.learning_phase():0}
    )
    adv_data_eval = K.get_session().run(
        adv_data,
        {data:rand_data_val, label:rand_label_val, batch_size:args.num_samples, K.learning_phase():0}
    )

    canvas = np.zeros([args.num_samples*28, 4*28])
    canvas[:, 0:28] = np.concatenate(rand_data_val).reshape([-1,28])
    canvas[:, 1*28:2*28] = np.concatenate(clean_reconst_eval).reshape([-1,28])
    canvas[:, 2*28:3*28] = np.concatenate(adv_data_eval).reshape([-1,28])
    canvas[:, 3*28:4*28] = np.concatenate(adv_reconst_eval).reshape([-1,28])
    plt.imshow(canvas, cmap='gray')
    plt.savefig("{}adv_compare.png".format(src_name))


