import cleverhans.utils
import cleverhans.utils_tf
import numpy as np
import tensorflow as tf
import cleverhans.attacks_tf
import tqdm
from cleverhans.attacks import SaliencyMapMethod
import keras.backend as K

import logging

def logits_stats(
    data, logits, sess,
    data_val,
):
    mean, var = tf.nn.moments(logits, axes=[0,1])
    result = sess.run([mean, var], {data:data_val, K.learning_phase():0})
    print "logits mean {} var {}".format(*result)

def soft_label_stats(
    data, label, model,
    data_val, label_val, num_samples
):
    ZERO = 1e-33
    sess = K.get_session()
    logits = model(data)
    preds = tf.nn.softmax(logits)
    mean_entropy = tf.losses.softmax_cross_entropy(
        onehot_labels = preds,
        logits = logits
    )

    indices = np.random.choice(len(data_val), num_samples, False)
    samples = data_val[indices]
    feed_dict = {data : samples, K.learning_phase() : 0}
    entropy_eval = sess.run(mean_entropy, feed_dict)
    print "mean entropy for softmax is {}".format(entropy_eval)

def score_mask_stats(
    data, logits,
    data_val, label_val, args,
):
    preds = tf.nn.softmax(logits)
    jacobian = cleverhans.attacks_tf.jacobian_graph(preds, data, 10)

    nb_classes = 10
    nb_features = 784
    index = 15

    x_origin = data_val[index]
    current_class = int(np.argmax(label_val[index]))
    target = 0
    feed_dict = {data : x_origin.reshape((1,28,28,1)), is_train: False}

    jacobian_val = np.zeros((nb_classes, nb_features), dtype=np.float32)
    for class_ind, grad in enumerate(jacobian):
        grad_eval = sess.run(grad, feed_dict)
        jacobian_val[class_ind] = np.reshape(grad_eval, (1, nb_features))
    other_classes = cleverhans.utils.other_classes(nb_classes, target)
    grad_others = np.sum(jacobian_val[other_classes, :], axis=0)
    grad_target = jacobian_val[target]
    target_sum = grad_target.reshape((1, nb_features)) + grad_target.reshape((nb_features, 1))
    other_sum = grad_others.reshape((1, nb_features)) + grad_others.reshape((nb_features, 1))

    pos_mask = ((target_sum > 0) & (other_sum < 0))
    score = pos_mask * (-other_sum * target_sum)

    print "at temperature {}".format(args.temp)
    print "total score of all pass mask {}".format(np.sum(score))

def jsma_eval(
    data, logits, sess,
    data_val, label_val,
    args,
):
    ''' model is using the logits or softmax result '''
    logging.info("here")
    nb_classes = 10
    grads = cleverhans.attacks_tf.jacobian_graph(tf.nn.softmax(logits), data, 10)
    logging.info("graph built")
    # Loop over the samples we want to perturb into adversarial examples
    num_samples = args.num_samples
    num_success = 0
    success_percent_perturb = 0
    for sample_ind in tqdm.tqdm(np.random.choice(len(data_val), num_samples, False)):
        sample = data_val[sample_ind:(sample_ind+1)]

        # We want to find an adversarial example for each possible target class
        # (i.e. all classes that differ from the label given in the dataset)
        current_class = int(np.argmax(label_val[sample_ind]))
        target_classes = cleverhans.utils.other_classes(nb_classes, current_class)
        target = target_classes[np.random.randint(nb_classes-1)]

        data_adv_eval, res, percent_perturb = cleverhans.attacks_tf.jsma(
            sess=sess,
            x=data,
            predictions=logits,
            grads=grads,
            sample=sample,
            target=target,
            theta=args.theta,
            gamma=args.gamma,
            clip_min=0,
            clip_max=255,
            feed={K.learning_phase() : 0}
        )
        # Update the arrays for later analysis
        if res == 1:
            num_success += 1
            success_percent_perturb += percent_perturb

    # Compute the number of adversarial examples that were successfully found
    print('Avg. rate of successful adv. examples {0:.4f}'.format(num_success /
                                                                float(num_samples)))

    # Compute the average distortion introduced for successful samples only
    print('Avg. rate of perturbed features for successful '
          'adversarial examples {0:.4f}'.format(success_percent_perturb /
                                                num_samples))

def iter_fgm_eval(
    data, label, model,
    data_val, label_val, num_samples
):
    sess = K.get_session()
    logits = model(data)
    # use true label
    eps = 0.3
    steps = 4
    preds = tf.nn.softmax(logits / 100)
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels = label,
        logits = logits,
    )
    grad, = tf.gradients(loss, data)
    scaled_signed_grad = eps * tf.sign(grad) / steps
    data_adv = tf.stop_gradient(data + scaled_signed_grad)
    data_adv = tf.clip_by_value(data_adv, 0, 1)

    # get data
    indices = np.random.choice(len(data_val), num_samples, False)
    samples = data_val[indices]
    labels = label_val[indices]

    for _ in range(steps):
        samples = sess.run(
            data_adv,
            {data : samples, label : labels, K.learning_phase() : 0}
        )
    preds_adv = sess.run(
        preds,
        {data : samples, label : labels, K.learning_phase() : 0}
    )
    acc = np.mean(
        np.argmax(preds_adv, 1) == np.argmax(labels,1)
    )
    print "iterative fgm sample err rate is: {}".format(1-acc)


def fgsm_eval(
    data, model,
    data_val, label_val, num_samples
):
    sess = K.get_session()
    logits = model(data)
    eps = 0.3
    preds = tf.nn.softmax(logits)
    # using model prediction as ground truth
    preds_max = tf.reduce_max(preds, 1, keep_dims=True)
    label = tf.to_float(tf.equal(preds, preds_max))

    loss = tf.losses.softmax_cross_entropy(
        onehot_labels = label,
        logits = logits,
    )
    grad, = tf.gradients(loss, data)
    ## add mean and std of gradient ##
    mean, var = tf.nn.moments(grad, axes=[0,1,2,3])

    signed_grad = tf.sign(grad)
    scaled_signed_grad = eps * signed_grad
    data_adv = tf.stop_gradient(data + scaled_signed_grad)
    data_adv = tf.clip_by_value(data_adv, 0, 1)

    # get data
    indices = np.random.choice(len(data_val), num_samples, False)
    samples = data_val[indices]
    labels = label_val[indices]

    samples_adv, mean_eval, var_eval = sess.run(
        [data_adv, mean, var],
        {
            data: samples,
            K.learning_phase():0
        }
    )
    logging.info("gradient mean is: {}, gradient var is: {}".format(mean_eval, var_eval))
    preds_adv = sess.run(preds, {data: samples_adv, K.learning_phase(): 0})
    acc = np.mean(
        np.argmax(preds_adv, 1) == np.argmax(labels,1)
    )
    logging.info("basic grad sample err rate is: {}".format(100*(1-acc)))

