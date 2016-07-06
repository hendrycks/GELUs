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
t_end = t_start + 60 * (3*60 + 53)      # certainly save 3 hours and 53 mins into computation
t_since_last_save = t_start

# training parameters
initial_learning_rate = .0001
training_epochs = 150
batch_size = 128

# architecture parameters
n_labels = 10
crop_length = 32
n_channels = 3

try:
    nonlinearity_name = sys.argv[1]       # 'relu', 'elu', 'gelu', 'zero_I'
    print('Chosen nonlinearity:', nonlinearity_name)
except:
    print('Defaulted to GELU since no nonlinearity specified through command line')
    nonlinearity_name = 'relu'

x = tf.placeholder(dtype=tf.float32, shape=[None, crop_length, crop_length, n_channels])
y = tf.placeholder(dtype=tf.int64, shape=[None])
is_training = tf.placeholder(tf.bool)


# make weights
W = {}

# we need to normalize each filter
out_represenations = []
for _ in range(64):
    u = tf.random_normal([3, 3, 3, 1])
    out_represenations.append(u[:, :, :, :] / tf.sqrt(tf.reduce_sum(tf.square(u[:, :, :, :])) + 1e-12))
W['1'] = tf.Variable(tf.reshape(tf.concat(3, out_represenations), [3, 3, 3, 64]))

out_represenations = []
for _ in range(64):
    u = tf.random_normal([3, 3, 64, 1])
    out_represenations.append(u[:, :, :, :] / tf.sqrt(tf.reduce_sum(tf.square(u[:, :, :, :])) + 1e-12))
W['2'] = tf.Variable(tf.reshape(tf.concat(3, out_represenations), [3, 3, 64, 64]))
# stack end

out_represenations = []
for _ in range(128):
    u = tf.random_normal([3, 3, 64, 1])
    out_represenations.append(u[:, :, :, :] / tf.sqrt(tf.reduce_sum(tf.square(u[:, :, :, :])) + 1e-12))
W['3'] = tf.Variable(tf.reshape(tf.concat(3, out_represenations), [3, 3, 64, 128]))

out_represenations = []
for _ in range(128):
    u = tf.random_normal([3, 3, 128, 1])
    out_represenations.append(u[:, :, :, :] / tf.sqrt(tf.reduce_sum(tf.square(u[:, :, :, :])) + 1e-12))
W['4'] = tf.Variable(tf.reshape(tf.concat(3, out_represenations), [3, 3, 128, 128]))
# stack end

out_represenations = []
for _ in range(256):
    u = tf.random_normal([3, 3, 128, 1])
    out_represenations.append(u[:, :, :, :] / tf.sqrt(tf.reduce_sum(tf.square(u[:, :, :, :])) + 1e-12))
W['5'] = tf.Variable(tf.reshape(tf.concat(3, out_represenations), [3, 3, 128, 256]))

out_represenations = []
for _ in range(256):
    u = tf.random_normal([3, 3, 256, 1])
    out_represenations.append(u[:, :, :, :] / tf.sqrt(tf.reduce_sum(tf.square(u[:, :, :, :])) + 1e-12))
W['6'] = tf.Variable(tf.reshape(tf.concat(3, out_represenations), [3, 3, 256, 256]))

out_represenations = []
for _ in range(256):
    u = tf.random_normal([3, 3, 256, 1])
    out_represenations.append(u[:, :, :, :] / tf.sqrt(tf.reduce_sum(tf.square(u[:, :, :, :])) + 1e-12))
W['7'] = tf.Variable(tf.reshape(tf.concat(3, out_represenations), [3, 3, 256, 256]))
# stack end

out_represenations = []
for _ in range(512):
    u = tf.random_normal([3, 3, 256, 1])
    out_represenations.append(u[:, :, :, :] / tf.sqrt(tf.reduce_sum(tf.square(u[:, :, :, :])) + 1e-12))
W['8'] = tf.Variable(tf.reshape(tf.concat(3, out_represenations), [3, 3, 256, 512]))

out_represenations = []
for _ in range(512):
    u = tf.random_normal([3, 3, 512, 1])
    out_represenations.append(u[:, :, :, :] / tf.sqrt(tf.reduce_sum(tf.square(u[:, :, :, :])) + 1e-12))
W['9'] = tf.Variable(tf.reshape(tf.concat(3, out_represenations), [3, 3, 512, 512]))

out_represenations = []
for _ in range(512):
    u = tf.random_normal([3, 3, 512, 1])
    out_represenations.append(u[:, :, :, :] / tf.sqrt(tf.reduce_sum(tf.square(u[:, :, :, :])) + 1e-12))
W['10'] = tf.Variable(tf.reshape(tf.concat(3, out_represenations), [3, 3, 512, 512]))
# stack end

out_represenations = []
for _ in range(512):
    u = tf.random_normal([3, 3, 512, 1])
    out_represenations.append(u[:, :, :, :] / tf.sqrt(tf.reduce_sum(tf.square(u[:, :, :, :])) + 1e-12))
W['11'] = tf.Variable(tf.reshape(tf.concat(3, out_represenations), [3, 3, 512, 512]))

out_represenations = []
for _ in range(512):
    u = tf.random_normal([3, 3, 512, 1])
    out_represenations.append(u[:, :, :, :] / tf.sqrt(tf.reduce_sum(tf.square(u[:, :, :, :])) + 1e-12))
W['12'] = tf.Variable(tf.reshape(tf.concat(3, out_represenations), [3, 3, 512, 512]))

out_represenations = []
for _ in range(512):
    u = tf.random_normal([3, 3, 512, 1])
    out_represenations.append(u[:, :, :, :] / tf.sqrt(tf.reduce_sum(tf.square(u[:, :, :, :])) + 1e-12))
W['13'] = tf.Variable(tf.reshape(tf.concat(3, out_represenations), [3, 3, 512, 512]))
# stack end

W['14'] = tf.Variable(tf.nn.l2_normalize(tf.random_normal([512, 512]), 0))
W['15'] = tf.Variable(tf.nn.l2_normalize(tf.random_normal([512, n_labels]), 0))

b = {
    '1': tf.Variable(tf.zeros([64])),
    '2': tf.Variable(tf.zeros([64])),
    '3': tf.Variable(tf.zeros([128])),
    '4': tf.Variable(tf.zeros([128])),
    '5': tf.Variable(tf.zeros([256])),
    '6': tf.Variable(tf.zeros([256])),
    '7': tf.Variable(tf.zeros([256])),
    '8': tf.Variable(tf.zeros([512])),
    '9': tf.Variable(tf.zeros([512])),
    '10': tf.Variable(tf.zeros([512])),
    '11': tf.Variable(tf.zeros([512])),
    '12': tf.Variable(tf.zeros([512])),
    '13': tf.Variable(tf.zeros([512])),
    '14': tf.Variable(tf.zeros([512])),
    '15': tf.Variable(tf.zeros([n_labels]))
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

    h1 = rho(tf.nn.conv2d(x, W['1'], strides=[1, 1, 1, 1], padding='SAME') + b['1'])
    h2 = rho(tf.nn.conv2d(h1, W['2'], strides=[1, 1, 1, 1], padding='SAME') + b['2'])
    max1 = tf.nn.max_pool(h2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    max1 = tf.cond(is_training, lambda: tf.nn.dropout(max1, 0.6), lambda: max1)

    h3 = rho(tf.nn.conv2d(max1, W['3'], strides=[1, 1, 1, 1], padding='SAME') + b['3'])
    h3 = tf.cond(is_training, lambda: tf.nn.dropout(h3, 0.6), lambda: h3)
    h4 = rho(tf.nn.conv2d(h3, W['4'], strides=[1, 1, 1, 1], padding='SAME') + b['4'])
    max2 = tf.nn.max_pool(h4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    max2 = tf.cond(is_training, lambda: tf.nn.dropout(max2, 0.6), lambda: max2)

    h5 = rho(tf.nn.conv2d(max2, W['5'], strides=[1, 1, 1, 1], padding='SAME') + b['5'])
    h5 = tf.cond(is_training, lambda: tf.nn.dropout(h5, 0.6), lambda: h5)
    h6 = rho(tf.nn.conv2d(h5, W['6'], strides=[1, 1, 1, 1], padding='SAME') + b['6'])
    h6 = tf.cond(is_training, lambda: tf.nn.dropout(h6, 0.6), lambda: h6)
    h7 = rho(tf.nn.conv2d(h6, W['7'], strides=[1, 1, 1, 1], padding='SAME') + b['7'])
    max3 = tf.nn.max_pool(h7, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    max3 = tf.cond(is_training, lambda: tf.nn.dropout(max3, 0.6), lambda: max3)

    h8 = rho(tf.nn.conv2d(max3, W['8'], strides=[1, 1, 1, 1], padding='SAME') + b['8'])
    h8 = tf.cond(is_training, lambda: tf.nn.dropout(h8, 0.6), lambda: h6)
    h9 = rho(tf.nn.conv2d(h8, W['9'], strides=[1, 1, 1, 1], padding='SAME') + b['9'])
    h9 = tf.cond(is_training, lambda: tf.nn.dropout(h9, 0.6), lambda: h9)
    h10 = rho(tf.nn.conv2d(h9, W['10'], strides=[1, 1, 1, 1], padding='SAME') + b['10'])
    max4 = tf.nn.max_pool(h10, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    max4 = tf.cond(is_training, lambda: tf.nn.dropout(max4, 0.6), lambda: max4)

    h11 = rho(tf.nn.conv2d(max4, W['11'], strides=[1, 1, 1, 1], padding='SAME') + b['11'])
    h11 = tf.cond(is_training, lambda: tf.nn.dropout(h11, 0.6), lambda: h11)
    h12 = rho(tf.nn.conv2d(h11, W['12'], strides=[1, 1, 1, 1], padding='SAME') + b['12'])
    h12 = tf.cond(is_training, lambda: tf.nn.dropout(h12, 0.6), lambda: h12)
    h13 = rho(tf.nn.conv2d(h12, W['13'], strides=[1, 1, 1, 1], padding='SAME') + b['13'])
    max5 = tf.nn.max_pool(h13, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # unroll layer
    max5 = tf.reshape(max5, [-1, W['14'].get_shape().as_list()[0]])
    max5 = tf.cond(is_training, lambda: tf.nn.dropout(max5, 0.5), lambda: max5)

    h14 = rho(tf.matmul(max5, W['14']) + b['14'])
    h14 = tf.cond(is_training, lambda: tf.nn.dropout(h14, 0.5), lambda: h14)
    return tf.matmul(h14, W['15']) + b['15']

logits = feedforward(x)
global_step = tf.Variable(0, trainable=False)
loss_ema = tf.Variable(2.3, trainable=False)
lr = tf.train.exponential_decay(initial_learning_rate, global_step, 50*390, 1/5., staircase=True)
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, y))
optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss, global_step=global_step)


def compute_error(y_hat, y_gold):       # y_gold does not have a one-hot representation
    return tf.not_equal(tf.argmax(y_hat, 1), y_gold)


with tf.Session() as sess:
    print('Loading data.')
    sess.run(tf.initialize_all_variables())
    # store future results with previous results
    if not os.path.exists("./data/"):
        os.makedirs("./data/")

    indicate_restored = 0
    saver = tf.train.Saver(max_to_keep=1)
    if os.path.exists("./data/cifar10_" + nonlinearity_name + ".ckpt"):
        saver.restore(sess, "./data/cifar10_" + nonlinearity_name + ".ckpt")
        print('Restored!')
        indicate_restored = 1
    else:
        for step_num in range(training_epochs*50000//batch_size, -1, -1):    # search downward due to aggressive saving
            if os.path.exists("./data/cifar10_" + nonlinearity_name + ".ckpt-"+str(step_num)):
                saver.restore(sess, "./data/cifar10_" + nonlinearity_name + ".ckpt-"+str(step_num))
                print('Restored! Epoch', step_num)
                indicate_restored = 1
                break

    if os.path.exists("./data/" + nonlinearity_name + "_history.p"):
        if indicate_restored:
            history = pickle.load(open("./data/" + nonlinearity_name + "_history.p", "rb"))
            max_key = 0
            for k in history.keys():
                max_key = max(max_key, int(k[-1]))
            key_str = str(max_key)
        else:
            history = pickle.load(open("./data/" + nonlinearity_name + "_history.p", "rb"))
            key_str = str(int(len(history) / 6) + 1)
            history["loss" + key_str] = []
            history["loss_val" + key_str] = []
            history["loss_val_best" + key_str] = 1000
            history["err_val" + key_str] = []
            history["err_test" + key_str] = 1
    else:
        history = {'loss1': [], 'test1': 1, 'loss_val1': [], 'loss_val_best1': 1000,
                   'err_val1': [], 'err_test1': 1}
        key_str = '1'

    X_train, Y_train, X_test, Y_test = load_data()

    num_batches = int(X_train.shape[0] / batch_size)
    checkpoint_every = int(batch_size/2.1)      # save training information 2 times per epoch
    # should_drop = False if nonlinearity_name == 'zero_I' else True
    print('Beginning training.')
    for epoch in range(sess.run(global_step)//num_batches, training_epochs):     # indicate_restored is probably needless here

        # train for an epoch
        for i in range(num_batches):
            loss_ema = tf.minimum(loss_ema, 10)    # don't get thrown off by early 1e12 losses
            offset = i * batch_size
            bx, by = X_train[offset:offset+batch_size, :, :, :], Y_train[offset:offset+batch_size]
            _, l = sess.run([optimizer, loss], feed_dict={x: bx, y: by, is_training: True})
            loss_ema = loss_ema * 0.95 + 0.05 * l

            if i % checkpoint_every == 0 and i > 0:
                # we feed in so large a dataset for visualization purposes
                perm = np.random.random_integers(50000 - 1, size=512)
                l_stable = sess.run(loss, feed_dict={x: X_train[perm, :, :, :], y: Y_train[perm], is_training: True})
                l_stable = loss_ema * 0.90 + 0.1 * l_stable
                history['loss' + key_str].append(sess.run(l_stable))

        # save results
        # l_val, logits_val = sess.run([loss, logits], feed_dict={x: X_val, y: Y_val,
        #                                                         is_training: False})
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

        if time.time() > t_since_last_save + 30 * 60 or time.time() > t_end or epoch >= training_epochs - 1:
            # save every 30 minutes or if it's been 3 hours and 40 minutes
            t_since_last_save = time.time()
            saver.save(sess, './data/cifar10_' + nonlinearity_name + ".ckpt", global_step=global_step)
            pickle.dump(history, open("./data/" + nonlinearity_name + "_history.p", "wb"))
        # done saving

        print('Epoch:', epoch, '| ema loss:', sess.run(loss_ema))

    # done
    global_step += 1   # now there will be no future iterations upon re-running this file
    saver.save(sess, './data/cifar10_' + nonlinearity_name + ".ckpt", global_step=global_step)
    pickle.dump(history, open("./data/" + nonlinearity_name + "_history.p", "wb"))
    print('Done. Test score:', history['err_test' + key_str])
