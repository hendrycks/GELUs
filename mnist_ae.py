# import MNIST data, Tensorflow, and other helpers
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/")
import tensorflow as tf
import numpy as np
import sys
import os
import pickle

# training parameters
training_epochs = 500
batch_size = 64

# architecture parameters
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

x = tf.placeholder(dtype=tf.float32, shape=[None, image_pixels])

W = {
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
    h2 = f(tf.matmul(h1, W['2']) + b['2'])
    h3 = f(tf.matmul(h2, W['3']) + b['3'])
    h4 = f(tf.matmul(h3, W['4']) + b['4'])
    h5 = f(tf.matmul(h4, W['5']) + b['5'])
    h6 = f(tf.matmul(h5, W['6']) + b['6'])
    h7 = f(tf.matmul(h6, W['7']) + b['7'])
    return tf.matmul(h7, W['8']) + b['8']

reconstruction = ae(x)
loss = tf.reduce_mean(tf.square(reconstruction - x))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# store future results with previous results
if not os.path.exists("./data/"):
    os.makedirs("./data/")

if os.path.exists("./data/mnist_ae_" + nonlinearity_name + ".p"):
    history = pickle.load(open("./data/mnist_ae_" + nonlinearity_name + ".p", "rb"))
    key_str = str(len(history)//3 + 1)
    history["lr" + key_str] = learning_rate
    history["train_loss" + key_str] = []
    history["test_loss" + key_str] = []
else:
    history = {'lr1': learning_rate, 'train_loss1': [], 'test_loss1': []}
    key_str = '1'


with tf.Session() as sess:
    print('Loading Data')
    X_train = np.concatenate((mnist.train.images, mnist.validation.images), axis=0)

    print('Beginning training')
    sess.run(tf.initialize_all_variables())

    num_batches = 60000 // batch_size
    save_every = num_batches//5        # save training information 3 times per epoch

    for epoch in range(training_epochs):
        # shuffle
        indices = np.arange(X_train.shape[0])
        np.random.shuffle(indices)
        X_train = X_train[indices]

        for i in range(num_batches):
            offset = i * batch_size
            _, l = sess.run([optimizer, loss], feed_dict={x: X_train[offset:offset+batch_size]})

            history["train_loss" + key_str].append(l)
            if i % save_every == 0:
                l = sess.run(loss, feed_dict={x: mnist.test.images})
                history["test_loss" + key_str].append(l)

        # print('Epoch', epoch + 1, 'Complete')

# save history
pickle.dump(history, open("./data/mnist_ae_" + nonlinearity_name + ".p", "wb"))
