import numpy as np
import theano as th
import theano.tensor as T
from scipy import linalg
import lasagne

class ZCA(object):
    def __init__(self, regularization=1e-5, x=None):
        self.regularization = regularization
        if x is not None:
            self.fit(x)

    def fit(self, x):
        s = x.shape
        x = x.copy().reshape((s[0],np.prod(s[1:])))
        m = np.mean(x, axis=0)
        x -= m
        sigma = np.dot(x.T,x) / x.shape[0]
        U, S, V = linalg.svd(sigma)
        tmp = np.dot(U, np.diag(1./np.sqrt(S+self.regularization)))
        tmp2 = np.dot(U, np.diag(np.sqrt(S+self.regularization)))
        self.ZCA_mat = th.shared(np.dot(tmp, U.T).astype(th.config.floatX))
        self.inv_ZCA_mat = th.shared(np.dot(tmp2, U.T).astype(th.config.floatX))
        self.mean = th.shared(m.astype(th.config.floatX))

    def apply(self, x):
        s = x.shape
        if isinstance(x, np.ndarray):
            return np.dot(x.reshape((s[0],np.prod(s[1:]))) - self.mean.get_value(), self.ZCA_mat.get_value()).reshape(s)
        elif isinstance(x, T.TensorVariable):
            return T.dot(x.flatten(2) - self.mean.dimshuffle('x',0), self.ZCA_mat).reshape(s)
        else:
            raise NotImplementedError("Whitening only implemented for numpy arrays or Theano TensorVariables")
            
    def invert(self, x):
        s = x.shape
        if isinstance(x, np.ndarray):
            return (np.dot(x.reshape((s[0],np.prod(s[1:]))), self.inv_ZCA_mat.get_value()) + self.mean.get_value()).reshape(s)
        elif isinstance(x, T.TensorVariable):
            return (T.dot(x.flatten(2), self.inv_ZCA_mat) + self.mean.dimshuffle('x',0)).reshape(s)
        else:
            raise NotImplementedError("Whitening only implemented for numpy arrays or Theano TensorVariables")

# T.nnet.relu has some issues with very large inputs, this is more stable
def relu(x):
    return T.maximum(x, 0)

def lrelu(x, a=0.1):
    return T.maximum(x, a*x)

def gelu(x):
    return 0.5 * x * (1 + T.tanh(T.sqrt(2 / np.pi) * (x + 0.044715 * T.pow(x, 3))))

def log_sum_exp(x, axis=1):
    m = T.max(x, axis=axis)
    return m+T.log(T.sum(T.exp(x-m.dimshuffle(0,'x')), axis=axis))

def adamax_updates(params, cost, lr=0.001, mom1=0.9, mom2=0.999):
    updates = []
    grads = T.grad(cost, params)
    for p, g in zip(params, grads):
        mg = th.shared(np.cast[th.config.floatX](p.get_value() * 0.))
        v = th.shared(np.cast[th.config.floatX](p.get_value() * 0.))
        if mom1>0:
            v_t = mom1*v + (1. - mom1)*g
            updates.append((v,v_t))
        else:
            v_t = g
        mg_t = T.maximum(mom2*mg, abs(g))
        g_t = v_t / (mg_t + 1e-6)
        p_t = p - lr * g_t
        updates.append((mg, mg_t))
        updates.append((p, p_t))
    return updates

def adam_updates(params, cost, lr=0.001, mom1=0.9, mom2=0.999):
    updates = []
    grads = T.grad(cost, params)
    t = th.shared(np.cast[th.config.floatX](1.))
    for p, g in zip(params, grads):
        v = th.shared(np.cast[th.config.floatX](p.get_value() * 0.))
        mg = th.shared(np.cast[th.config.floatX](p.get_value() * 0.))
        v_t = mom1*v + (1. - mom1)*g
        mg_t = mom2*mg + (1. - mom2)*T.square(g)
        v_hat = v_t / (1. - mom1 ** t)
        mg_hat = mg_t / (1. - mom2 ** t)
        g_t = v_hat / T.sqrt(mg_hat + 1e-8)
        p_t = p - lr * g_t
        updates.append((v, v_t))
        updates.append((mg, mg_t))
        updates.append((p, p_t))
    updates.append((t, t+1))
    return updates
    
def softmax_loss(p_true, output_before_softmax):
    output_before_softmax -= T.max(output_before_softmax, axis=1, keepdims=True)
    if p_true.ndim==2:
        return T.mean(T.log(T.sum(T.exp(output_before_softmax),axis=1)) - T.sum(p_true*output_before_softmax, axis=1))
    else:
        return T.mean(T.log(T.sum(T.exp(output_before_softmax),axis=1)) - output_before_softmax[T.arange(p_true.shape[0]),p_true])

class GlobalAvgLayer(lasagne.layers.Layer):
    def __init__(self, incoming, **kwargs):
        super(GlobalAvgLayer, self).__init__(incoming, **kwargs)
    def get_output_for(self, input, **kwargs):
        return T.mean(input, axis=(2,3))
    def get_output_shape_for(self, input_shape):
        return input_shape[:2]
