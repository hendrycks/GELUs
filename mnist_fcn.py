# import MNIST data, Tensorflow, and other helpers
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/")
import tensorflow as tf
import numpy as np
import sys
import os
import pickle

# training parameters
training_epochs = 50
batch_size = 128

# architecture parameters
n_hidden = 128
n_labels = 10
image_pixels = 28 * 28

try:
    nonlinearity_name = sys.argv[1]       # 'relu', 'elu', 'gelu', 'silu'
except:
    print('Defaulted to gelu since no nonlinearity specified through command line')
    nonlinearity_name = 'gelu'

try:
    learning_rate = float(sys.argv[2])       # 0.001, 0.0001, 0.00001
except:
    print('Defaulted to a learning rate of 0.001')
    learning_rate = 1e-3

try:
    p = float(sys.argv[3])       # 1 or 0.5
except:
    print('Defaulted to to a dropout keep probability of 1.0')
    p = 1.

x = tf.placeholder(dtype=tf.float32, shape=[None, image_pixels])
y = tf.placeholder(dtype=tf.int64, shape=[None])
is_training = tf.placeholder(tf.bool)


W = {
    '1': tf.Variable(tf.nn.l2_normalize(tf.random_normal([image_pixels, n_hidden]), 0)),
    '2': tf.Variable(tf.nn.l2_normalize(tf.random_normal([n_hidden, n_hidden]), 0)),
    '3': tf.Variable(tf.nn.l2_normalize(tf.random_normal([n_hidden, n_hidden]), 0)),
    '4': tf.Variable(tf.nn.l2_normalize(tf.random_normal([n_hidden, n_hidden]), 0)),
    '5': tf.Variable(tf.nn.l2_normalize(tf.random_normal([n_hidden, n_hidden]), 0)),
    '6': tf.Variable(tf.nn.l2_normalize(tf.random_normal([n_hidden, n_hidden]), 0)),
    '7': tf.Variable(tf.nn.l2_normalize(tf.random_normal([n_hidden, n_hidden]), 0)),
    '8': tf.Variable(tf.nn.l2_normalize(tf.random_normal([n_hidden, n_hidden]), 0)),
    '9': tf.Variable(tf.nn.l2_normalize(tf.random_normal([n_hidden, n_labels]), 0))
}

b = {
    '1': tf.Variable(tf.zeros([n_hidden])),
    '2': tf.Variable(tf.zeros([n_hidden])),
    '3': tf.Variable(tf.zeros([n_hidden])),
    '4': tf.Variable(tf.zeros([n_hidden])),
    '5': tf.Variable(tf.zeros([n_hidden])),
    '6': tf.Variable(tf.zeros([n_hidden])),
    '7': tf.Variable(tf.zeros([n_hidden])),
    '8': tf.Variable(tf.zeros([n_hidden])),
    '9': tf.Variable(tf.zeros([n_labels]))
}

def feedforward(x):
    if nonlinearity_name == 'relu':
        f = tf.nn.relu
    elif nonlinearity_name == 'elu':
        f = tf.nn.elu
    elif nonlinearity_name == 'gelu':
        # def gelu(x):
        #     return tf.mul(x, tf.erfc(-x / tf.sqrt(2.)) / 2.)
        # f = gelu
        def gelu_fast(_x):
            return 0.5 * _x * (1 + tf.tanh(tf.sqrt(2 / np.pi) * (_x + 0.044715 * tf.pow(_x, 3))))
        f = gelu_fast
    elif nonlinearity_name == 'silu':
        def silu(_x):
            return _x * tf.sigmoid(_x)
        f = silu
    # elif nonlinearity_name == 'soi':
    #     def soi_map(x):
    #         u = tf.random_uniform(tf.shape(x))
    #         mask = tf.to_float(tf.less(u, (1 + tf.erf(x / tf.sqrt(2.))) / 2.))
    #         return tf.cond(is_training, lambda: tf.mul(mask, x),
    #                        lambda: tf.mul(x, tf.erfc(-x / tf.sqrt(2.)) / 2.))
    #     f = soi_map

    else:
        raise NameError("Need 'relu', 'elu', 'gelu', or 'silu' for nonlinearity_name")

    h1 = f(tf.matmul(x, W['1']) + b['1'])
    h1 = tf.cond(is_training, lambda: tf.nn.dropout(h1, p), lambda: h1)
    h2 = f(tf.matmul(h1, W['2']) + b['2'])
    h2 = tf.cond(is_training, lambda: tf.nn.dropout(h2, p), lambda: h2)
    h3 = f(tf.matmul(h2, W['3']) + b['3'])
    h3 = tf.cond(is_training, lambda: tf.nn.dropout(h3, p), lambda: h3)
    h4 = f(tf.matmul(h3, W['4']) + b['4'])
    h4 = tf.cond(is_training, lambda: tf.nn.dropout(h4, p), lambda: h4)
    h5 = f(tf.matmul(h4, W['5']) + b['5'])
    h5 = tf.cond(is_training, lambda: tf.nn.dropout(h5, p), lambda: h5)
    h6 = f(tf.matmul(h5, W['6']) + b['6'])
    h6 = tf.cond(is_training, lambda: tf.nn.dropout(h6, p), lambda: h6)
    h7 = f(tf.matmul(h6, W['7']) + b['7'])
    h7 = tf.cond(is_training, lambda: tf.nn.dropout(h7, p), lambda: h7)
    h8 = f(tf.matmul(h7, W['8']) + b['8'])
    h8 = tf.cond(is_training, lambda: tf.nn.dropout(h8, p), lambda: h8)
    return tf.matmul(h8, W['9']) + b['9']

logits = feedforward(x)
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

compute_error = tf.reduce_mean(tf.to_float(tf.not_equal(tf.argmax(logits, 1), y)))

# store future results with previous results
if not os.path.exists("./data/"):
    os.makedirs("./data/")

if os.path.exists("./data/mnist_fcn_" + nonlinearity_name + ".p"):
    history = pickle.load(open("./data/mnist_fcn_" + nonlinearity_name + ".p", "rb"))
    key_str = str(len(history)//8 + 1)
    history["lr" + key_str] = learning_rate
    history["dropout" + key_str] = p
    history["train_loss" + key_str] = []
    history["val_loss" + key_str] = []
    history["test_loss" + key_str] = []
    history["train_err" + key_str] = []
    history["val_err" + key_str] = []
    history["test_err" + key_str] = []
else:
    history = {
        "lr1": learning_rate, "dropout1": p,
        'train_loss1': [], 'val_loss1': [], 'test_loss1': [],
        'train_err1': [], 'val_err1': [], 'test_err1': []
    }
    key_str = '1'


with tf.Session() as sess:
    print('Beginning training')
    sess.run(tf.initialize_all_variables())

    num_batches = mnist.train.num_examples // batch_size
    save_every = num_batches//3        # save training information 3 times per epoch

    for epoch in range(training_epochs):
        for i in range(num_batches):
            bx, by = mnist.train.next_batch(batch_size)

            if p < 1-1e-5:   # we want to know how the full network is being optimized instead of the reduced version
                l, err = sess.run([loss, compute_error], feed_dict={x: bx, y: by, is_training: False})

            _, l_drop, err_drop = sess.run([optimizer, loss, compute_error], feed_dict={x: bx, y: by,
                                                                                        is_training: True})

            if p < 1-1e-5:   # we want to know how the full network is being optimized instead of the reduced version
                history["train_loss" + key_str].append(l)
                history["train_err" + key_str].append(err)
            else:
                history["train_loss" + key_str].append(l_drop)
                history["train_err" + key_str].append(err_drop)

            # save
            if i % save_every == 0:
                l, err = sess.run([loss, compute_error],
                                  feed_dict={x: mnist.validation.images,
                                             y: mnist.validation.labels,
                                             is_training: False})
                history["val_loss" + key_str].append(l)
                history["val_err" + key_str].append(err)

                l, err = sess.run([loss, compute_error],
                                  feed_dict={x: mnist.test.images,
                                             y: mnist.test.labels,
                                             is_training: False})
                history["test_loss" + key_str].append(l)
                history["test_err" + key_str].append(err)

        # print('Epoch', epoch + 1, 'Complete')

# save history
pickle.dump(history, open("./data/mnist_fcn_" + nonlinearity_name + ".p", "wb"))
