import numpy as np
import tensorflow as tf
import pickle
import sys
import io
import os

try:
    nonlinearity_name = sys.argv[1]       # 'relu', 'elu', 'gelu', or 'silu'
except:
    print('Defaulted to gelu since no nonlinearity specified through command line')
    nonlinearity_name = 'gelu'

try:
    learning_rate = float(sys.argv[2])       # 0.001, 0.0001, 0.00001
except:
    print('Defaulted to a learning rate of 0.001')
    learning_rate = 1e-3

p = 0.8

#
# Begin Twitter Helper Functions
#

def embeddings_to_dict(filename):
    '''
    :param filename: the file name of the word embeddings | file is assumed
    to follow this format: "word[tab]dimension 1[space]dimension 2[space]...[space]dimension 50"
    :return: a dictionary with keys that are words and values that are the embedding of a word
    '''
    with io.open(filename, 'r', encoding='utf-8') as f:
        word_vecs = {}
        for line in f:
            line = line.strip('\n').split()
            word_vecs[line[0]] = np.array([float(s) for s in line[1:]])

    return word_vecs


def data_to_mat(filename, vocab, tag_to_number, window_size=1, start_symbol=u'UUUNKKK',
                one_hot=False, return_labels=True):
    '''
    :param filename: the filename of a training, development, devtest, or test set
    :param vocab: a list of strings, one for each embedding (the keys of a dictionary)
    :param tag_to_number: a dictionary of tags to predict and a numerical encoding of those tags;
    with this, we will predict numbers instead of strings
    :param window_size: the context window size for the left and right; thus we have 2*window_size + 1
    words considered at a time
    :param start_symbol: since the <s> symbol has no embedding given, chose a symbol in the vocab
    to replace <s>. Common choices are u'UUUNKKK' or u'</s>'
    :return: a n x (window_size*2 + 1) matrix containing context windows and the center word
    represented as strings; n is the number of examples. ALSO return a n x |tag_to_number|
    matrix of labels for the n examples with a one-hot (1-of-k) encoding
    '''
    with io.open(filename, 'r', encoding='utf-8') as f:
        x, tweet_words, y = [], [], []
        start = True
        for line in f:
            line = line.strip('\n')

            if len(line) == 0:              # if end of tweet
                tweet_words.extend([u'</s>'] * window_size)

                # ensure tweet words are in vocab; if not, map to "UUUNKKK"

                tweet_words = [w if w in vocab else u'UUUNKKK' for w in tweet_words]

                # from this tweet, add the training tasks to dataset
                # the tags were already added to y
                for i in range(window_size, len(tweet_words) - window_size):
                    x.append(tweet_words[i-window_size:i+window_size+1])

                tweet_words = []
                start = True
                continue

            # if before end
            word, label = line.split('\t')

            if start:
                tweet_words.extend([start_symbol] * window_size)
                start = False

            tweet_words.append(word)

            if return_labels is True:
                if one_hot is True:
                    label_one_hot = len(tag_to_number) * [0]
                    label_one_hot[tag_to_number[label]] += 1

                    y.append(label_one_hot)
                else:
                    y.append(tag_to_number[label])

    return np.array(x), np.array(y)


def word_list_to_embedding(words, embeddings, embedding_dimension=50):
    '''
    :param words: an n x (2*window_size + 1) matrix from data_to_mat
    :param embeddings: an embedding dictionary where keys are strings and values
    are embeddings; the output from embeddings_to_dict
    :param embedding_dimension: the dimension of the values in embeddings; in this
    assignment, embedding_dimension=50
    :return: an n x ((2*window_size + 1)*embedding_dimension) matrix where each entry of the
    words matrix is replaced with its embedding
    '''
    m, n = words.shape
    words = words.reshape((-1))

    return np.array([embeddings[w] for w in words], dtype=np.float32).reshape(m, n*embedding_dimension)

#
# End Twitter Helper Functions
#

window_size = 1

# note that we encode the tags with numbers for later convenience
tag_to_number = {
    u'N': 0, u'O': 1, u'S': 2, u'^': 3, u'Z': 4, u'L': 5, u'M': 6,
    u'V': 7, u'A': 8, u'R': 9, u'!': 10, u'D': 11, u'P': 12, u'&': 13, u'T': 14,
    u'X': 15, u'Y': 16, u'#': 17, u'@': 18, u'~': 19, u'U': 20, u'E': 21, u'$': 22,
    u',': 23, u'G': 24
}

embeddings = embeddings_to_dict('./data/Tweets/embeddings-twitter.txt')
vocab = embeddings.keys()

# we replace <s> with </s> since it has no embedding, and </s> is a better embedding than UNK
xt, yt = data_to_mat('./data/Tweets/tweets-train.txt', vocab, tag_to_number, window_size=window_size,
                     start_symbol=u'</s>')
xdev, ydev = data_to_mat('./data/Tweets/tweets-dev.txt', vocab, tag_to_number, window_size=window_size,
                         start_symbol=u'</s>')
xdtest, ydtest = data_to_mat('./data/Tweets/tweets-devtest.txt', vocab, tag_to_number, window_size=window_size,
                             start_symbol=u'</s>')

data = {
    'x_train': xt, 'y_train': yt,
    'x_dev': xdev, 'y_dev': ydev,
    'x_test': xdtest, 'y_test': ydtest
}

num_epochs = 30
num_tags = 25
hidden_size = 256
batch_size = 16
embedding_dimension = 50
example_size = (2*window_size + 1)*embedding_dimension
num_examples = data['y_train'].shape[0]
num_batches = num_examples//batch_size

graph = tf.Graph()
with graph.as_default():
    x = tf.placeholder(tf.float32, [None, example_size])
    y = tf.placeholder(tf.int64, [None])
    is_training = tf.placeholder(tf.bool)

    w1 = tf.Variable(tf.nn.l2_normalize(tf.random_normal([example_size, hidden_size]), 0))
    b1 = tf.Variable(tf.zeros([hidden_size]))
    w2 = tf.Variable(tf.nn.l2_normalize(tf.random_normal([hidden_size, hidden_size]), 0))
    b2 = tf.Variable(tf.zeros([hidden_size]))
    w_out = tf.Variable(tf.nn.l2_normalize(tf.random_normal([hidden_size, num_tags]), 0))
    b_out = tf.Variable(tf.zeros([num_tags]))

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

    def model(data_feed):
        h1 = f(tf.matmul(data_feed, w1) + b1)
        h1 = tf.cond(is_training, lambda: tf.nn.dropout(h1, p), lambda: h1)
        h2 = f(tf.matmul(h1, w2) + b2)
        h2 = tf.cond(is_training, lambda: tf.nn.dropout(h2, p), lambda: h2)
        return tf.matmul(h2, w_out) + b_out

    logits = model(x)
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, y))

    # pick optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    compute_error = tf.reduce_mean(tf.to_float(tf.not_equal(tf.argmax(logits, 1), y)))

# store future results with previous results
if not os.path.exists("./data/"):
    os.makedirs("./data/")

if os.path.exists("./data/twitter_pos_" + nonlinearity_name + ".p"):
    history = pickle.load(open("./data/twitter_pos_" + nonlinearity_name + ".p", "rb"))
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

with tf.Session(graph=graph) as sess:
    print('Beginning training')
    sess.run(tf.initialize_all_variables())

    save_every = num_batches//5        # save training information 5 times per epoch

    # train
    for epoch in range(num_epochs):
        # shuffle data every epoch
        indices = np.arange(num_examples)
        np.random.shuffle(indices)
        data['x_train'] = data['x_train'][indices]
        data['y_train'] = data['y_train'][indices]

        for i in range(num_batches):
            offset = i * batch_size

            bx = word_list_to_embedding(data['x_train'][offset:offset + batch_size, :],
                                        embeddings, embedding_dimension)
            by = data['y_train'][offset:offset + batch_size]

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

            if i % save_every == 0:
                l, err = sess.run([loss, compute_error],
                                  feed_dict={x: word_list_to_embedding(data['x_dev'], embeddings, embedding_dimension),
                                             y: data['y_dev'], is_training: False})
                history["val_loss" + key_str].append(l)
                history["val_err" + key_str].append(err)

                l, err = sess.run([loss, compute_error],
                                  feed_dict={x: word_list_to_embedding(data['x_test'], embeddings, embedding_dimension),
                                             y: data['y_test'], is_training: False})
                history["test_loss" + key_str].append(l)
                history["test_err" + key_str].append(err)

            # print('Epoch', epoch + 1, 'Complete')

# save history
pickle.dump(history, open("./data/twitter_pos_" + nonlinearity_name + ".p", "wb"))
