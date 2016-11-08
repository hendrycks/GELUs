import tensorflow as tf
import numpy as np
import h5py
import pickle
import sys
import io
import os

# training parameters
training_epochs = 30
batch_size = 64

# architecture parameters
n_hidden = 2048
n_labels = 39   # 39 phones
n_coeffs = 26
n_context_frames = 11   # 5 + 1 + 5
p = 0.5             # keep rate

try:
    nonlinearity_name = sys.argv[1]       # 'relu', 'elu', or 'gelu'
except:
    print('Defaulted to gelu since no nonlinearity specified through command line')
    nonlinearity_name = 'gelu'

try:
    learning_rate = float(sys.argv[2])       # 0.001, 0.0001, 0.00001
except:
    print('Defaulted to a learning rate of 0.001')
    learning_rate = 1e-3


def enumerate_context(i, sentence, num_frames):
    r = range(i-num_frames, i+num_frames+1)
    r = [x if x>=0 else 0 for x in r]
    r = [x if x<len(sentence) else len(sentence)-1 for x in r]
    return sentence[r]

def add_context(sentence, num_frames=11):
    # [sentence_length, coefficients] -> [sentence_length, num_frames, coefficients]

    assert num_frames % 2 == 1, "Number of frames must be odd (since left + 1 + right, left = right)"

    if num_frames == 1:
        return sentence

    context_sent = []

    for i in range(0, len(sentence)):
        context_sent.append([context for context in enumerate_context(i, sentence, (num_frames-1)//2)])

    return np.array(context_sent).reshape((-1, num_frames*n_coeffs))

print('Making graph')
graph = tf.Graph()
with graph.as_default():
    x = tf.placeholder(dtype=tf.float32, shape=[None, n_coeffs*n_context_frames])
    y = tf.placeholder(dtype=tf.int64, shape=[None])
    is_training = tf.placeholder(tf.bool)

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
    else:
        raise NameError("Need 'relu', 'elu', 'gelu', for nonlinearity_name")

    W = {
        '1': tf.Variable(tf.nn.l2_normalize(tf.random_normal([n_context_frames*n_coeffs, n_hidden]), 0)),
        '2': tf.Variable(tf.nn.l2_normalize(tf.random_normal([n_hidden, n_hidden]), 0)),
        '3': tf.Variable(tf.nn.l2_normalize(tf.random_normal([n_hidden, n_hidden]), 0)),
        '4': tf.Variable(tf.nn.l2_normalize(tf.random_normal([n_hidden, n_hidden]), 0)),
        '5': tf.Variable(tf.nn.l2_normalize(tf.random_normal([n_hidden, n_hidden]), 0)),
        '6': tf.Variable(tf.nn.l2_normalize(tf.random_normal([n_hidden, n_hidden]), 0)),
        '7': tf.Variable(tf.nn.l2_normalize(tf.random_normal([n_hidden, n_labels]), 0)),
    }
    b = {
        '1': tf.Variable(tf.zeros([n_hidden])),
        '2': tf.Variable(tf.zeros([n_hidden])),
        '3': tf.Variable(tf.zeros([n_hidden])),
        '4': tf.Variable(tf.zeros([n_hidden])),
        '5': tf.Variable(tf.zeros([n_hidden])),
        '6': tf.Variable(tf.zeros([n_hidden])),
        '7': tf.Variable(tf.zeros([n_labels]))
    }

    def feedforward(x):
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

        return tf.matmul(h5, W['6']) + b['6']

    logits = feedforward(x)
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, y))

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    compute_error = tf.reduce_mean(tf.to_float(tf.not_equal(tf.argmax(logits, 1), y)))


# store future results with previous results
if not os.path.exists("./data/"):
    os.makedirs("./data/")

if os.path.exists("./data/timit_fcn_" + nonlinearity_name + ".p"):
    history = pickle.load(open("./data/timit_fcn_" + nonlinearity_name + ".p", "rb"))
    key_str = str(len(history)//7 + 1)
    history["lr" + key_str] = learning_rate
    history["train_loss" + key_str] = []
    history["val_loss" + key_str] = []
    history["test_loss" + key_str] = []
    history["train_err" + key_str] = []
    history["val_err" + key_str] = []
    history["test_err" + key_str] = []
else:
    history = {
        "lr1": learning_rate,
        'train_loss1': [], 'val_loss1': [], 'test_loss1': [],
        'train_err1': [], 'val_err1': [], 'test_err1': []
    }
    key_str = '1'

print('Loading Data')
data = h5py.File("./data/train.h5")
X_train = data['X'][()]
Y_train = data['y'][()]
train_idxs = data['start_idx'][()]

train_mean = np.mean(X_train, axis=(0,1))
train_std = np.std(X_train, axis=(0,1))
X_train -= train_mean
X_train /= (train_std + 1e-11)

data = h5py.File("./data/dev.h5")
X_val = data['X'][()] - train_mean
Y_val = data['y'][()]
val_idxs = data['start_idx'][()]
X_val -= train_mean
X_val /= (train_std + 1e-11)

data = h5py.File("./data/core.h5")
X_test = data['X'][()] - train_mean
Y_test = data['y'][()]
test_idxs = data['start_idx'][()]
X_test -= train_mean
X_test /= (train_std + 1e-11)
del data
print('Number of training examples', X_train.shape[0])
print('Number of validation examples', X_val.shape[0])
print('Number of testing examples', X_test.shape[0])


with tf.Session(graph=graph) as sess:
    sess.run(tf.initialize_all_variables())

    num_batches = X_train.shape[0] // batch_size
    save_every = num_batches//5        # save training information 5 times per epoch

    for epoch in range(training_epochs):
        # shuffle data
        indices = np.arange(X_train.shape[0])
        np.random.shuffle(indices)
        X_train = X_train[indices]
        Y_train = Y_train[indices]
        train_idxs = train_idxs[indices]

        for i in range(num_batches):
            offset = i * batch_size
            _bx, mask_x, _by = X_train[offset:offset+batch_size], train_idxs[offset:offset+batch_size], Y_train[offset:offset+batch_size]

            bx, by = [], []
            for j in range(_bx.shape[0]):
                sentence_frames = add_context(_bx[j][mask_x[j]:])
                bx.append(sentence_frames)
                by.append(_by[j][mask_x[j]:])

            bx, by = np.concatenate(bx), np.concatenate(by)

            if p < 1-1e-5:   # we want to know how the full network is being optimized instead of the reduced version
                l, err = sess.run([loss, compute_error], feed_dict={x: bx, y: by, is_training: False})

            _, l_drop, err_drop = sess.run([optimizer, loss, compute_error], feed_dict={x: bx, y: by,
                                                                                        is_training: True})

            if p < 1-1e-5:
                history["train_loss" + key_str].append(l)
                history["train_err" + key_str].append(err)
            else:
                history["train_loss" + key_str].append(l_drop)
                history["train_err" + key_str].append(err_drop)

            if i % save_every == 0:
                err_total = 0
                loss_total = 0
                for j in range(X_test.shape[0]//batch_size):
                    offset = j * batch_size
                    _bx, mask_x, _by = X_test[offset:offset+batch_size], test_idxs[offset:offset+batch_size], Y_test[offset:offset+batch_size]

                    bx, by = [], []
                    for k in range(_bx.shape[0]):
                        sentence_frames = add_context(_bx[k][mask_x[k]:])
                        bx.append(sentence_frames)
                        by.append(_by[k][mask_x[k]:])

                    bx, by = np.concatenate(bx), np.concatenate(by)

                    l, err = sess.run([loss, compute_error], feed_dict={x: bx, y: by, is_training: False})
                    loss_total += l
                    err_total += err
                history["test_loss" + key_str].append(loss_total/(X_test.shape[0]//batch_size))
                history["test_err" + key_str].append(err_total/(X_test.shape[0]//batch_size))

                err_total = 0
                loss_total = 0
                for j in range(X_val.shape[0]//batch_size):
                    offset = j * batch_size
                    _bx, mask_x, _by = X_val[offset:offset+batch_size], val_idxs[offset:offset+batch_size], Y_val[offset:offset+batch_size]

                    bx, by = [], []
                    for k in range(_bx.shape[0]):
                        sentence_frames = add_context(_bx[k][mask_x[k]:])
                        bx.append(sentence_frames)
                        by.append(_by[k][mask_x[k]:])

                    bx, by = np.concatenate(bx), np.concatenate(by)

                    l, err = sess.run([loss, compute_error], feed_dict={x: bx, y: by, is_training: False})
                    loss_total += l
                    err_total += err
                history["val_loss" + key_str].append(loss_total/(X_val.shape[0]//batch_size))
                history["val_err" + key_str].append(err_total/(X_val.shape[0]//batch_size))

        print('Epoch', epoch+1, 'Complete.', 'Val loss', history["val_loss" + key_str][-1], 'Val error', history["val_err" + key_str][-1])

# save history
pickle.dump(history, open("./data/timit_fcn_" + nonlinearity_name + ".p", "wb"))
