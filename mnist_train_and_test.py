# import MNIST data, Tensorflow, and other helpers
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/")
import tensorflow as tf
import numpy as np
import sys
import os
import pickle

# training parameters
learning_rate = 0.001
training_epochs = 50
batch_size = 128

# architecture parameters
n_hidden = 128
n_labels = 10
image_pixels = 28 * 28

try:
    nonlinearity_name = sys.argv[1]       # 'relu', 'elu', 'gelu', 'zero_I'
except:
    print('Defaulted to relu since no nonlinearity specified through command line')
    nonlinearity_name = 'gelu'

x = tf.placeholder(dtype=tf.float32, shape=[None, image_pixels])
y = tf.placeholder(dtype=tf.int64, shape=[None])
is_training = tf.placeholder(tf.bool)

if nonlinearity_name != 'zero_I':
    p = 0.5
else:
    p = 1

W = {
    '1': tf.Variable(tf.nn.l2_normalize(tf.random_normal([image_pixels, n_hidden]), 0)),
    '2': tf.Variable(tf.nn.l2_normalize(tf.random_normal([n_hidden, n_hidden]), 0)),
    '3': tf.Variable(tf.nn.l2_normalize(tf.random_normal([n_hidden, n_hidden]), 0)),
    '4': tf.Variable(tf.nn.l2_normalize(tf.random_normal([n_hidden, n_hidden]), 0)),
    '5': tf.Variable(tf.nn.l2_normalize(tf.random_normal([n_hidden, n_hidden]), 0)),
    '6': tf.Variable(tf.nn.l2_normalize(tf.random_normal([n_hidden, n_hidden]), 0)),
    '7': tf.Variable(tf.nn.l2_normalize(tf.random_normal([n_hidden, n_hidden]), 0)),
    '8': tf.Variable(tf.nn.l2_normalize(tf.random_normal([n_hidden, n_labels]), 0))
}

b = {
    '1': tf.Variable(tf.zeros([n_hidden])),
    '2': tf.Variable(tf.zeros([n_hidden])),
    '3': tf.Variable(tf.zeros([n_hidden])),
    '4': tf.Variable(tf.zeros([n_hidden])),
    '5': tf.Variable(tf.zeros([n_hidden])),
    '6': tf.Variable(tf.zeros([n_hidden])),
    '7': tf.Variable(tf.zeros([n_hidden])),
    '8': tf.Variable(tf.zeros([n_labels]))
}

def feedforward(x):
    if nonlinearity_name == 'relu':
        rho = tf.nn.relu
    elif nonlinearity_name == 'elu':
        rho = tf.nn.elu
    elif nonlinearity_name == 'gelu':
        def gelu(x):
            return tf.mul(x, tf.erfc(-x / tf.sqrt(2.)) / 2.)
        rho = gelu
    elif nonlinearity_name == 'zero_I':
        def zero_identity_map(x):
            u = tf.random_uniform(tf.shape(x))
            mask = tf.to_float(tf.less(u, (1 + tf.erf(x / tf.sqrt(2.))) / 2.))
            return tf.cond(is_training, lambda: tf.mul(mask, x),
                           lambda: tf.mul(x, tf.erfc(-x / tf.sqrt(2.)) / 2.))
        rho = zero_identity_map

    else:
        raise NameError("Need 'relu', 'elu', 'gelu', or 'zero_I' for nonlinearity_name")

    h1 = rho(tf.matmul(x, W['1']) + b['1'])
    h1 = tf.cond(is_training, lambda: tf.nn.dropout(h1, p), lambda: h1)
    h2 = rho(tf.matmul(h1, W['2']) + b['2'])
    h2 = tf.cond(is_training, lambda: tf.nn.dropout(h2, p), lambda: h2)
    h3 = rho(tf.matmul(h2, W['3']) + b['3'])
    h3 = tf.cond(is_training, lambda: tf.nn.dropout(h3, p), lambda: h3)
    h4 = rho(tf.matmul(h3, W['4']) + b['4'])
    h4 = tf.cond(is_training, lambda: tf.nn.dropout(h4, p), lambda: h4)
    h5 = rho(tf.matmul(h4, W['5']) + b['5'])
    h5 = tf.cond(is_training, lambda: tf.nn.dropout(h5, p), lambda: h5)
    h6 = rho(tf.matmul(h5, W['6']) + b['6'])
    h6 = tf.cond(is_training, lambda: tf.nn.dropout(h6, p), lambda: h6)
    h7 = rho(tf.matmul(h6, W['7']) + b['7'])
    h7 = tf.cond(is_training, lambda: tf.nn.dropout(h7, p), lambda: h7)
    return tf.matmul(h7, W['8']) + b['8']


logits = feedforward(x)
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# store future results with previous results
if not os.path.exists("./data/"):
    os.makedirs("./data/")

if os.path.exists("./data/" + nonlinearity_name + "_history.p"):
    history = pickle.load(open("./data/" + nonlinearity_name + "_history.p", "rb"))
    key_str = str(len(history)//3 + 1)
    history["loss" + key_str] = []
    history["loss_val" + key_str] = []
    history["test" + key_str] = 1
else:
    history = {'loss1': [], 'loss_val1': [], 'test1': 1}
    key_str = '1'


with tf.Session() as sess:
    print('Beginning training')
    sess.run(tf.initialize_all_variables())

    num_batches = int(mnist.train.num_examples / batch_size)
    save_every = int(batch_size/3.1)      # save training information 3 times per epoch
    loss_ema = 2.3  # - log(2.3)
    for epoch in range(training_epochs):
        for i in range(num_batches):
            bx, by = mnist.train.next_batch(batch_size)
            _, l = sess.run([optimizer, loss], feed_dict={x: bx, y: by, is_training: True})
            loss_ema = loss_ema * 0.95 + 0.05 * l

            if i % save_every == 0:
                # for comparability be sure to have both
                # computed deterministically or stochastically!
                l_stable = sess.run(loss,
                                    feed_dict={x: mnist.train.images,
                                               y: mnist.train.labels,
                                               is_training: False})
                history['loss' + key_str].append(l_stable)
                l_val = sess.run(loss, feed_dict={x: mnist.validation.images,
                                                  y: mnist.validation.labels,
                                                  is_training: False})
                history['loss_val' + key_str].append(l_val)

        print('Epoch:', epoch, '|', 'ema of loss for epoch:', loss_ema)

    # computation finished---now test
    compute_error = tf.not_equal(tf.argmax(logits, 1), y)
    history['test' + key_str] = sess.run(tf.reduce_mean(tf.to_float(compute_error)),
                                         feed_dict={x: mnist.test.images,
                                                    y: mnist.test.labels,
                                                    is_training: False})
    print('Test error (%):', 100*history['test' + key_str])

# save history
pickle.dump(history, open("./data/" + nonlinearity_name + "_history.p", "wb"))

