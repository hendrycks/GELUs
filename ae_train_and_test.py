# Scroll down to the "% save_every == 0" condition and modify the code there if
# the experiment is too slow

# import MNIST data, Tensorflow, and other helpers
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/")
import tensorflow as tf
import numpy as np
import sys
import os
import pickle

# training parameters
training_epochs = 300
batch_size = 64

# architecture parameters
image_pixels = 28 * 28

try:
    nonlinearity_name = sys.argv[1]       # 'relu', 'elu', 'gelu', 'zero_I'
except:
    print('Defaulted to ReLU since no nonlinearity specified through command line')
    nonlinearity_name = 'relu'

try:
    learning_rate = float(sys.argv[2])         # 1e-5, ... , 1e-2
except:
    print('Defaulted to a 0.001 learning rate since no lr specified through command line')
    learning_rate = 0.001

lr_str = "%E" % learning_rate       # 0.01 becomes 1.000000E-02; lrs from 1e-5, ... , 1e-2 have consistent string length

x = tf.placeholder(dtype=tf.float32, shape=[None, image_pixels])

W = {
    # '1': tf.Variable(tf.random_normal([image_pixels, n_hidden])/tf.sqrt(image_pixels + n_hidden * 0.5)),
    # '2': tf.Variable(tf.random_normal([n_hidden, n_hidden])/tf.sqrt(n_hidden * 0.5 + n_hidden * 0.5)),
    # '3': tf.Variable(tf.random_normal([n_hidden, n_hidden])/tf.sqrt(n_hidden * 0.5 + n_hidden * 0.5)),
    # '4': tf.Variable(tf.random_normal([n_hidden, n_hidden])/tf.sqrt(n_hidden * 0.5 + n_hidden * 0.5)),
    # '5': tf.Variable(tf.random_normal([n_hidden, n_hidden])/tf.sqrt(n_hidden * 0.5 + n_hidden * 0.5)),
    # '6': tf.Variable(tf.random_normal([n_hidden, n_hidden])/tf.sqrt(n_hidden * 0.5 + n_hidden * 0.5)),
    # '7': tf.Variable(tf.random_normal([n_hidden, n_hidden])/tf.sqrt(n_hidden * 0.5 + n_hidden * 0.5)),
    # '8': tf.Variable(tf.random_normal([n_hidden, n_labels])/tf.sqrt(n_hidden * 0.5 + n_labels))
    # '1': tf.Variable(tf.random_normal([image_pixels, n_hidden]) / tf.sqrt(image_pixels)),
    # '2': tf.Variable(tf.random_normal([n_hidden, n_hidden]) / tf.sqrt(n_hidden)),
    # '3': tf.Variable(tf.random_normal([n_hidden, n_hidden]) / tf.sqrt(n_hidden)),
    # '4': tf.Variable(tf.random_normal([n_hidden, n_hidden]) / tf.sqrt(n_hidden)),
    # '5': tf.Variable(tf.random_normal([n_hidden, n_hidden]) / tf.sqrt(n_hidden)),
    # '6': tf.Variable(tf.random_normal([n_hidden, n_hidden]) / tf.sqrt(n_hidden)),
    # '7': tf.Variable(tf.random_normal([n_hidden, n_hidden]) / tf.sqrt(n_hidden)),
    # '8': tf.Variable(tf.random_normal([n_hidden, n_labels]) / tf.sqrt(n_hidden))
    '1': tf.Variable(tf.nn.l2_normalize(tf.random_normal([image_pixels, 1000]), 0)),
    '2': tf.Variable(tf.nn.l2_normalize(tf.random_normal([1000, 500]), 0)),
    '3': tf.Variable(tf.nn.l2_normalize(tf.random_normal([500, 250]), 0)),
    '4': tf.Variable(tf.nn.l2_normalize(tf.random_normal([250, 30]), 0)),
    '5': tf.Variable(tf.nn.l2_normalize(tf.random_normal([30, 250]), 0)),
    '6': tf.Variable(tf.nn.l2_normalize(tf.random_normal([250, 500]), 0)),
    '7': tf.Variable(tf.nn.l2_normalize(tf.random_normal([500, 1000]), 0)),
    '8': tf.Variable(tf.nn.l2_normalize(tf.random_normal([1000, image_pixels]), 0))
}

b = {
    '1': tf.Variable(tf.zeros([1000])),
    '2': tf.Variable(tf.zeros([500])),
    '3': tf.Variable(tf.zeros([250])),
    '4': tf.Variable(tf.zeros([30])),
    '5': tf.Variable(tf.zeros([250])),
    '6': tf.Variable(tf.zeros([500])),
    '7': tf.Variable(tf.zeros([1000])),
    '8': tf.Variable(tf.zeros([image_pixels]))
}

def ae(x):
    if nonlinearity_name == 'relu':
        rho = tf.nn.relu
    elif nonlinearity_name == 'elu':
        rho = tf.nn.elu
    elif nonlinearity_name == 'gelu':
        def gelu(x):
            return tf.mul(x, tf.erfc(-x/tf.sqrt(2.))/2.)
        rho = gelu
    elif nonlinearity_name == 'zero_I':
        def zero_identity_map(x):
            u = tf.random_uniform(tf.shape(x))
            mask = tf.to_float(tf.less(u, (1 + tf.erf(x / tf.sqrt(2.))) / 2.))
            return tf.mul(mask, x)
        rho = zero_identity_map
    else:
        raise NameError("Need 'relu', 'elu', 'gelu', or 'zero_I' for nonlinearity_name")

    h1 = rho(tf.matmul(x, W['1']) + b['1'])
    h2 = rho(tf.matmul(h1, W['2']) + b['2'])
    h3 = rho(tf.matmul(h2, W['3']) + b['3'])
    h4 = rho(tf.matmul(h3, W['4']) + b['4'])
    h5 = rho(tf.matmul(h4, W['5']) + b['5'])
    h6 = rho(tf.matmul(h5, W['6']) + b['6'])
    h7 = rho(tf.matmul(h6, W['7']) + b['7'])
    return tf.matmul(h7, W['8']) + b['8']


reconstruction = ae(x)
loss = tf.reduce_mean(tf.square(reconstruction - x))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# store future results with previous results
if not os.path.exists("./data/"):
    os.makedirs("./data/")

if os.path.exists("./data/" + nonlinearity_name + "_history.p"):
    history = pickle.load(open("./data/" + nonlinearity_name + "_history.p", "rb"))

    # determine current trial for particular learning rate by looking at keys
    biggest_key_num = 0
    for k in history.keys():
        if lr_str[-1] == k[16]:      # checking if exponents are the same
            biggest_key_num = max(int(k[18:]), biggest_key_num)
    key_str = str(biggest_key_num + 1)

    history["loss_" + lr_str + '_' + key_str] = []
    history["test_" + lr_str + '_' + key_str] = []
else:
    history = {'loss_'+lr_str+'_1': [], 'test_'+lr_str+'_1': []}
    key_str = '1'

# noinspection PyInterpreter
with tf.Session() as sess:
    print('Loading Data')
    # data has a 55000 5000 train/val split
    X_train = np.concatenate((mnist.train.images, mnist.validation.images), axis=0)
    num_examples = 60000

    print('Beginning training')
    sess.run(tf.initialize_all_variables())

    num_batches = int(num_examples / batch_size)
    save_every = int(batch_size/2.1)      # save training information 2 times per epoch
    ema_loss = 2.3
    for epoch in range(training_epochs):
        for i in range(num_batches):
            offset = i * batch_size
            _, l = sess.run([optimizer, loss], feed_dict={x: X_train[offset:offset+batch_size, :]})
            ema_loss = 0.95 * ema_loss + 0.05 * l

            if i % save_every == 0:
                # done for a stable curve for presentation purposes
                # MODIFY THIS SECTION IF YOU WANT TO RUN THIS EXPERIMENT MORE QUICKLY
                perm = np.random.random_integers(60000 - 1, size=20000)
                l_stable = sess.run(loss, feed_dict={x: X_train[perm, :]})
                history['loss_' + lr_str + '_' + key_str].append(l_stable)

        # epoch complete: now display results
        print('Epoch:', epoch, '|', 'average loss for epoch:', ema_loss)

        # save test results once per epoch since there are so many epochs
        l_test = sess.run(loss, feed_dict={x: mnist.test.images})
        history['test_' + lr_str + '_' + key_str].append(l_test)

# save history
pickle.dump(history, open("./data/" + nonlinearity_name + "_history.p", "wb"))
