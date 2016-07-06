# import CIFAR10 data, Tensorflow, and other helpers
import tensorflow as tf
import sys
import os
import pickle
import numpy as np
from six.moves import urllib
import tarfile
from load_cifar10 import load_data
import time

# cluster computing helper
t_start = time.time()
t_end = t_start + 60 * (3*60 + 40)      # certainly save 3 hours and 40 mins into computation
t_since_last_save = t_start

# training parameters
initial_learning_rate = 0.001
training_epochs = 200
batch_size = 64
p = 0.5

# architecture parameters
n_labels = 10
crop_length = 32
n_channels = 3

try:
    nonlinearity_name = sys.argv[1]       # 'relu', 'elu', 'gelu', 'zero_I'
    print('Chosen nonlinearity:', nonlinearity_name)
except:
    print('Defaulted to GELU since no nonlinearity specified through command line')
    nonlinearity_name = 'gelu'

x = tf.placeholder(dtype=tf.float32, shape=[None, crop_length, crop_length, n_channels])
y = tf.placeholder(dtype=tf.int64, shape=[None])
is_training = tf.placeholder(tf.bool)
apply_dropout = tf.placeholder(tf.bool)

# make weights
W = {}

# we need to normalize each filter
out_represenations_1 = []
for _ in range(32):
    u = tf.random_normal([3, 3, 3, 1])
    out_represenations_1.append(u[:, :, :, :] / tf.sqrt(tf.reduce_sum(tf.square(u[:, :, :, :])) + 1e-12))
W['1'] = tf.Variable(tf.reshape(tf.concat(3, out_represenations_1),
                                [3, 3, 3, 32]))
#stack end

out_represenations_2 = []
for _ in range(32):
    u = tf.random_normal([3, 3, 32, 1])
    out_represenations_2.append(u[:, :, :, :] / tf.sqrt(tf.reduce_sum(tf.square(u[:, :, :, :])) + 1e-12))
W['2'] = tf.Variable(tf.reshape(tf.concat(3, out_represenations_2),
                                [3, 3, 32, 32]))

out_represenations_3 = []
for _ in range(64):
    u = tf.random_normal([3, 3, 32, 1])
    out_represenations_3.append(u[:, :, :, :] / tf.sqrt(tf.reduce_sum(tf.square(u[:, :, :, :])) + 1e-12))
W['3'] = tf.Variable(tf.reshape(tf.concat(3, out_represenations_3),
                                [3, 3, 32, 64]))

out_represenations_4 = []
for _ in range(64):
    u = tf.random_normal([3, 3, 64, 1])
    out_represenations_4.append(u[:, :, :, :] / tf.sqrt(tf.reduce_sum(tf.square(u[:, :, :, :])) + 1e-12))
W['4'] = tf.Variable(tf.reshape(tf.concat(3, out_represenations_4),
                                [3, 3, 64, 64]))

W['5'] = tf.Variable(tf.nn.l2_normalize(tf.random_normal([256, 512]), 0))
W['6'] = tf.Variable(tf.nn.l2_normalize(tf.random_normal([512, 256]), 0))
W['7'] = tf.Variable(tf.nn.l2_normalize(tf.random_normal([256, n_labels]), 0))

b = {
    '1': tf.Variable(tf.zeros([32])),
    '2': tf.Variable(tf.zeros([32])),
    '3': tf.Variable(tf.zeros([64])),
    '4': tf.Variable(tf.zeros([64])),
    '5': tf.Variable(tf.zeros([512])),
    '6': tf.Variable(tf.zeros([256])),
    '7': tf.Variable(tf.zeros([n_labels]))
}


def feedforward(x):
    def gelu(x):
        return tf.mul(x, tf.erfc(-x / tf.sqrt(2.)) / 2.)
    if nonlinearity_name == 'relu':
        rho = tf.nn.relu
    elif nonlinearity_name == 'elu':
        rho = tf.nn.elu
    elif nonlinearity_name == 'gelu':
        rho = gelu
    elif nonlinearity_name == 'zero_I':
        def zero_identity_map(x):
            u = tf.random_uniform(tf.shape(x))
            mask = tf.to_float(tf.less(u, (1 + tf.erf(x / tf.sqrt(2.))) / 2.))
            return tf.mul(mask, x)

        rho = zero_identity_map
    else:
        raise NameError("Need 'relu', 'elu', 'gelu', or 'zero_I' for nonlinearity_name")

    h1 = rho(tf.nn.conv2d(x, W['1'], strides=[1, 2, 2, 1], padding='SAME') + b['1'])
    h2 = rho(tf.nn.conv2d(h1, W['2'], strides=[1, 2, 2, 1], padding='SAME') + b['2'])
    max1 = tf.nn.max_pool(h2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    max1 = tf.cond(tf.logical_and(apply_dropout, is_training),
                   lambda: tf.nn.dropout(max1, 1 - (1-p) / 2.), lambda: max1)

    h3 = rho(tf.nn.conv2d(max1, W['3'], strides=[1, 1, 1, 1], padding='SAME') + b['3'])
    h4 = rho(tf.nn.conv2d(h3, W['4'], strides=[1, 1, 1, 1], padding='SAME') + b['4'])
    max2 = tf.nn.max_pool(h4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    max2 = tf.cond(tf.logical_and(apply_dropout, is_training),
                   lambda: tf.nn.dropout(max2, 1 - (1-p)/2.), lambda: max2)

    # unroll layer
    max2 = tf.reshape(max2, [-1, W['5'].get_shape().as_list()[0]])

    h5 = rho(tf.matmul(max2, W['5']) + b['5'])
    h5 = tf.cond(tf.logical_and(apply_dropout, is_training),
                 lambda: tf.nn.dropout(h5, p), lambda: h5)
    h6 = rho(tf.matmul(h5, W['6']) + b['6'])
    return tf.matmul(h6, W['7']) + b['7']

logits = feedforward(x)
global_step = tf.Variable(0, trainable=False)
lr = tf.train.exponential_decay(initial_learning_rate,
                                global_step, 75, 1/3., staircase=True)
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, y))
optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)


def compute_error(y_hat, y_gold):       # y_gold does not have a one-hot representation
    return tf.not_equal(tf.argmax(y_hat, 1), y_gold)


# store future results with previous results
if not os.path.exists("./data/"):
    os.makedirs("./data/")

if os.path.exists("./data/" + nonlinearity_name + "_history.p"):
    history = pickle.load(open("./data/" + nonlinearity_name + "_history.p", "rb"))
    key_str = str(int(len(history)/6) + 1)
    history["loss" + key_str] = []
    history["loss_val" + key_str] = []
    history["loss_val_best" + key_str] = 1000
    history["err_val" + key_str] = []
    history["err_test" + key_str] = 1
else:
    history = {'loss1': [], 'test1': 1, 'loss_val1': [], 'loss_val_best1': 1000,
               'err_val1': [], 'err_test1': 1}
    key_str = '1'


with tf.Session() as sess:
    saver = tf.train.Saver(max_to_keep=1)
    if os.path.exists("./data/cifar10_" + nonlinearity_name + ".ckpt"):
        saver.restore(sess, "./data/cifar10_" + nonlinearity_name + ".ckpt")

    X_train, Y_train, X_test, Y_test = load_data()
    sess.run(tf.initialize_all_variables())

    num_batches = int(X_train.shape[0] / batch_size)
    checkpoint_every = int(batch_size/2.1)      # save training information 2 times per epoch
    should_drop = False if nonlinearity_name == 'zero_I' else True
    loss_ema = 2.3  # -log(0.1)
    for epoch in range(sess.run(global_step), training_epochs):

        # train for an epoch
        for i in range(num_batches):
            loss_ema = min(loss_ema, 10)    # don't get thrown off by initial big values
            offset = i * batch_size
            bx, by = X_train[offset:offset+batch_size, :, :, :], Y_train[offset:offset+batch_size]
            _, l = sess.run([optimizer, loss], feed_dict={x: bx, y: by,
                                                          is_training: True, apply_dropout: should_drop})
            loss_ema = loss_ema * 0.95 + 0.05 * l

            if i % checkpoint_every == 0 and i > 0:
                # we feed in so large a dataset for visualization purposes
                perm = np.random.random_integers(45000 - 1, size=1000)
                l_stable = sess.run(loss, feed_dict={x: X_train[perm, :, :, :], y: Y_train[perm],
                                                                is_training: True, apply_dropout: should_drop})
                l_stable = loss_ema * 0.90 + 0.1 * l_stable
                history['loss' + key_str].append(l_stable)

        # save results
        global_step += 1
        # l_val, logits_val = sess.run([loss, logits], feed_dict={x: X_val, y: Y_val,
        #                                                         is_training: False, apply_dropout: False})
        # err_val = sess.run(tf.reduce_mean(tf.to_float(compute_error(logits_val, Y_val))))
        # history['loss_val' + key_str].append(l_val)
        # history['err_val' + key_str].append(err_val)

        # if l_val < history['loss_val_best' + key_str] and global_step > 5:
        #     history['loss_val_best' + key_str] = l_val
        #
        #     # evaluate model on test set
        #     l_test, logits_test = sess.run([loss, logits], feed_dict={x: X_test, y: Y_test, is_training: False})
        #     err_test = sess.run(compute_error(logits_test, Y_test))
        #     history['err_test' + key_str] = err_test
        #
        #     saver.save(sess, 'cifar10_best_' + nonlinearity_name + ".ckpt", global_step=global_step)
        #     pickle.dump(history, open("./data/" + nonlinearity_name + "_history.p", "wb"))
        #
        # if time.time() > t_since_last_save + 30 * 60 or time.time() > t_end:
        #     # save every 30 minutes or if it's been 3 hours and 40 minutes
        #     t_since_last_save = time.time()
        #     saver.save(sess, 'cifar10_' + nonlinearity_name + ".ckpt", global_step=global_step)
        #     pickle.dump(history, open("./data/" + nonlinearity_name + "_history.p", "wb"))
        # # done saving

        print('Epoch:', epoch, '| ema loss:', loss_ema)

    # done
    global_step = training_epochs + 1   # now there will be no future iterations upon re-running this file
    # saver.save(sess, 'cifar10_' + nonlinearity_name + ".ckpt", global_step=global_step)
    pickle.dump(history, open("./data/" + nonlinearity_name + "_history.p", "wb"))
    print('Done. Test score:', history['err_test' + key_str])
