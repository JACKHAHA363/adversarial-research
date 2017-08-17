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
from tqdm import tqdm
logger = logging.getLogger()
logger.setLevel(logging.INFO)

######### parse arg #########################
p = argparse.ArgumentParser("exp on vae to detect adv samples")
p.add_argument('--sources', nargs='*', type=str)
p.add_argument('--vae', default='./vae', type=str)
p.add_argument('--num-samples', default=10, type=int)
p.add_argument('--theta', default=1, type=float)
p.add_argument('--gamma', default=0.005, type=float)
p.add_argument('--data-dir',
               default='/mnt/data-1/yuchen.lu/mnist_gzip/')
args = p.parse_args()

#############################################
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

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
    vae_loss = VAELoss(model, enc, data)
    clean_vae_eval = K.get_session().run(
        vae_loss,
        {data:rand_data_val, label:rand_label_val, K.learning_phase():0}
    )
    logging.info("{} clean has LB {}, reconst loss {}, kl loss {}".format(
        src_name,
        clean_vae_eval[0],
        clean_vae_eval[1],
        clean_vae_eval[2]
    ))
    reconst = model(data)
    clean_reconst_eval = K.get_session().run(
        reconst,
        {data:rand_data_val, label:rand_label_val, K.learning_phase():0}
    )

    # generate adv samples
    adv_data_eval = np.zeros(rand_data_val.shape)
    num_success = 0
    logits = src_model(data)
    for i in tqdm(range(len(rand_data_val))):
        sample = rand_data_val[i:(i+1)]
        curr_class = int(np.argmax(rand_label_val[i]))
        tar_classes = cleverhans.utils.other_classes(10, curr_class)
        target = tar_classes[np.random.randint(9)]
        adv_sample, res, percent_pert = cleverhans.attacks_tf.jsma(
            sess=K.get_session(),
            x=data,
            predictions=logits,
            grads=cleverhans.attacks_tf.jacobian_graph(logits, data, 10),
            sample=sample,
            target=target,
            gamma=args.gamma,
            theta=args.theta,
            clip_min=0,
            clip_max=1,
            feed={K.learning_phase():0}
        )
        adv_data_eval[i:(i+1), :, :, :] = adv_sample
        if res == 1:
            num_success += 1
    logging.info("rate of success {}".format(float(num_success) /
                                             args.num_samples))
    adv_vae_eval = K.get_session().run(
        vae_loss,
        {data:adv_data_eval, label:rand_label_val, K.learning_phase():0}
    )
    logging.info("{} adv has LB {}, reconst loss {}, kl loss {}".format(
        src_name,
        adv_vae_eval[0],
        adv_vae_eval[1],
        adv_vae_eval[2]
    ))
    adv_reconst_eval = K.get_session().run(
        reconst,
        {data:adv_data_eval, label:rand_label_val, K.learning_phase():0}
    )

    canvas = np.zeros([args.num_samples*28, 4*28])
    canvas[:, 0:28] = np.concatenate(rand_data_val).reshape([-1,28])
    canvas[:, 1*28:2*28] = np.concatenate(clean_reconst_eval).reshape([-1,28])
    canvas[:, 2*28:3*28] = np.concatenate(adv_data_eval).reshape([-1,28])
    canvas[:, 3*28:4*28] = np.concatenate(adv_reconst_eval).reshape([-1,28])
    plt.imshow(canvas, cmap='gray')
    plt.savefig("{}adv_compare.png".format(src_name))


