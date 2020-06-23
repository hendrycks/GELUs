# Gaussian Error Linear Units (GELUs)
This software allows users to reproduce the results in Gaussian Error Linear Units (GELUs), Dan Hendrycks and Kevin Gimpel 2016.

# GELU Approximations
The `sigmoid(1.702 * x) * x` approximation is fast but is somewhat inaccurate. Meanwhile `0.5 * x * (1 + tanh(x * 0.7978845608 * (1 + 0.044715 * x * x)))` is slower but more accurate.

However, exact versions are now available in pytorch, so approximations are no longer necessary for suitable speed.

# Execution
Please install Tensorflow, Lasagne, and Python 3+.

## Citation

If you find this useful in your research, please consider citing:

    @article{hendrycks2016gelu,
      title={Gaussian Error Linear Units (GELUs)},
      author={Hendrycks, Dan and Gimpel, Kevin},
      journal={arXiv preprint arXiv:1606.08415},
      year={2016}
    }
