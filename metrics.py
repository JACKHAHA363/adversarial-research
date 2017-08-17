import tensorflow as tf

ZERO = 1e-10
class Accuracy(object):
    # a moving average of accuracy that can be reset
    def __init__(self, label, preds, batch_size):
        self.num_inst = tf.Variable(0, dtype=tf.float32, trainable=False, name='num_inst')
        self.num_correct = tf.Variable(0, dtype=tf.float32, trainable=False, name='num_correct')
        batch_correct = tf.reduce_sum(tf.to_float(tf.equal(
            tf.arg_max(preds, 1),
            tf.arg_max(label, 1),
        )))
        self.update_ops = [
            tf.assign_add(self.num_inst, tf.constant(batch_size, tf.float32)),
            tf.assign_add(self.num_correct, batch_correct)
        ]
        self.reset_ops = [
            tf.assign(self.num_inst, tf.constant(0, tf.float32)),
            tf.assign(self.num_correct, tf.constant(0, tf.float32)),
        ]
        self.acc = self.num_correct / (self.num_inst + ZERO)

    def get_result(self):
        return self.acc
    def reset(self):
        return self.reset_ops
    def update(self):
        return self.update_ops
