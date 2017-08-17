import cleverhans.utils_tf as utils_tf
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import keras
import keras.backend as K
import math
import time
from metrics import Accuracy
import os
import argparse
import numpy as np

import logging

def get_data(data_dir):
    mnist = input_data.read_data_sets(data_dir, one_hot=True, reshape=False)
    data_train = np.vstack((mnist.train.images, mnist.validation.images))
    label_train = np.vstack((mnist.train.labels, mnist.validation.labels))
    data_val = mnist.test.images
    label_val = mnist.test.labels
    return (data_train, label_train, data_val, label_val)

def parse_args():
    p = argparse.ArgumentParser("a mnist train script")
    p.add_argument('--model', default='B', type=str)
    p.add_argument('--data-dir', default='/mnt/data-1/yuchen.lu/mnist_gzip/')
    p.add_argument('--batch-size', default=100, type=int)
    p.add_argument('--lr', default=0.001, type=float)
    p.add_argument('--num-epoch', default=1, type=int)
    p.add_argument('--model-prefix', default=None, type=str)
    p.add_argument('--temp', default=10.0, type=float)
    p.add_argument('--adv', default='false', type=str)
    p.add_argument('--dk-label', default=None, type=str)
    p.add_argument('--distill-temp', default=1.0, type=float)
    p.add_argument('--soft-weight', default=0, type=float)
    p.add_argument('--verbose', default='false', type=str)
    p.add_argument('--tbdir', default='./logs', type=str)
    args = p.parse_args()
    return args

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def model_train(
    data, label, model,
    data_train, label_train,
    data_val, label_val,
    soft_label_train,
    soft_label_val,
    args,
):
    sess = K.get_session()
    # build the loss and learning graph
    logits = model(data)
    preds = tf.nn.softmax(logits)
    acc = Accuracy(label, preds, args.batch_size)
    ce_hard = tf.losses.softmax_cross_entropy(
        onehot_labels = label,
        logits = logits / args.temp,
    )
    if str2bool(args.adv) is True:
        logging.info("train with adversarial samples from itself")
        from cleverhans.attacks_tf import fgsm
        data_adv = fgsm(x=data, predictions=preds, clip_min=0, clip_max=1)
        logits_adv = model(data_adv)
        ce_hard_adv = tf.losses.softmax_cross_entropy(
            onehot_labels = label,
            logits = logits_adv / args.temp,
        )
        loss_hard = 0.5*(ce_hard + ce_hard_adv)
    else:
        loss_hard = ce_hard

    if soft_label_train is not None:
        soft_label = tf.placeholder(tf.float32, shape=(None, 10))
        loss_soft = tf.losses.softmax_cross_entropy(
            onehot_labels = soft_label,
            logits = logits / args.distill_temp,
        )
        if args.soft_weight == 1:
            loss = loss_soft
        elif args.soft_weight == 0:
            loss = loss_hard
        else:
            loss = args.soft_weight * loss_soft + (1-args.soft_weight) * loss_hard
    else:
        loss = loss_hard

    lr = tf.placeholder(tf.float32)
    opt = tf.train.AdamOptimizer(learning_rate=lr)
    train_step = opt.minimize(loss)

    # tensorboard writer
    loss_tb = tf.summary.scalar('loss', loss)
    tb_proto = tf.summary.merge_all()
    writer = tf.summary.FileWriter(args.tbdir, sess.graph)
    step = 0

    # initialize
    utils_tf.initialize_uninitialized_global_variables(sess)
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
                K.learning_phase(): 1,
                lr : args.lr
            }
            if soft_label_train is not None:
                feed_dict[soft_label] = soft_label_train[start:end]
            run_list = [loss, tb_proto] + acc.update()
            run_list.append(train_step)
            run_result = sess.run(
                run_list,
                feed_dict=feed_dict,
            )
            loss_eval = run_result[0]
            tb_proto_eval = run_result[1]
            step += 1
            writer.add_summary(tb_proto_eval, step)
        train_acc = sess.run(acc.get_result())
        # eval on validation
        val_acc = model_eval(
            data=data, label=label,
            data_val=data_val, label_val=label_val,
            metric=acc, args=args,
        )
        cur = time.time()
        if str2bool(args.verbose):
            logging.info("Epoch take {} seconds, train acc {}, val acc {}".format(cur-prev, train_acc * 100, val_acc * 100))
        prev = cur
    writer.close()

    if args.model_prefix is not None:
        save_model(model, args.model_prefix)
    logging.info("at temperature {}, train err {}, val err {}".format(args.temp,
                                                               (1-train_acc)*100,
                                                               (1-val_acc)*100))
    return True

def save_model(model, prefix):
    dst_dir = os.path.dirname(prefix)
    if os.path.isdir(dst_dir) is False:
        os.mkdi(dst_dir)
    weight_path = prefix + ".param"
    keras.models.save_model(model, weight_path)

def save_vae(enc, dec, prefix):
    save_model(enc, prefix + "_enc")
    save_model(dec, prefix + "_dec")

def load_model(prefix):
    weight_path = prefix + ".param"
    model = keras.models.load_model(weight_path)
    return model

def load_vae(prefix):
    from keras.layers import Input, Lambda
    from keras.models import Model
    enc = load_model(prefix + "_enc")
    dec = load_model(prefix + "_dec")
    inputs = Input(shape=(28,28,1))
    [z_mean, z_log_var] = enc(inputs)
    # sampling
    def sampling(args):
        (mean, log_var) = args
        eps = K.random_normal(shape=K.shape(mean))
        tmp = mean + K.exp(log_var / 2) * eps
        return tmp
    z = Lambda(sampling)([z_mean, z_log_var])
    outputs = dec(z)
    return Model(inputs, outputs), enc, dec

def model_predict(data, preds, data_val, args):
    sess = K.get_session()
    nb_batches = int(math.ceil(float(len(data_val)) / args.batch_size))
    preds_evals = []
    for batch in range(nb_batches):
        # Compute batch start and end indices
        start, end = utils_tf.batch_indices(batch, len(data_val), args.batch_size)
        # prepare feed dict
        feed_dict = {
            data: data_val[start:end],
            K.learning_phase() : 0,
        }
        preds_eval=sess.run(preds, feed_dict) # (batch_size, 10)
        preds_evals.append(preds_eval)
    return np.concatenate(preds_evals)

def model_eval(data, label, data_val, label_val, metric, args, fd=None):
    sess = K.get_session()
    sess.run(metric.reset())
    nb_batches = int(math.ceil(float(len(data_val)) / args.batch_size))
    for batch in range(nb_batches):
        # Compute batch start and end indices
        start, end = utils_tf.batch_indices(batch, len(data_val), args.batch_size)
        # prepare feed dict
        feed_dict = {
            data: data_val[start:end],
            label: label_val[start:end],
            K.learning_phase() : 0,
        }
        if fd is not None:
            feed_dict.update(fd)
        run_result = sess.run(
            metric.update(),
            feed_dict=feed_dict,
        )
    result = sess.run(metric.get_result())
    return result



