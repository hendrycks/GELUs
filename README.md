# Bridging Nonlinearities and Stochastic Regularizers with Gaussian Error Linear Units
This software allows users to reproduce the results in Bridging Nonlinearities and Stochastic Regularizers with Gaussian Error Linear Units, Dan Hendrycks and Kevin Gimpel 2016.

# GELU Approximations
The `sigmoid(1.702 * x) * x` approximation is fast but is somewhat inaccurate. Meanwhile `0.5 * x * (1 + tanh(x * 0.7978845608 * (1 + 0.044715 * x * x)))` is slower but more accurate.

# Execution
Please install Tensorflow, Lasagne, and Python 3+.
