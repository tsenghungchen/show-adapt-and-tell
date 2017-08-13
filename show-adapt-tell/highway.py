import tensorflow as tf

# highway layer that borrowed from https://github.com/carpedm20/lstm-char-cnn-tensorflow
def highway(input_, size, layer_size=1, bias=-2, f=tf.nn.relu):
    """Highway Network (cf. http://arxiv.org/abs/1505.00387).

    t = sigmoid(Wy + b)
    z = t * g(Wy + b) + (1 - t) * y
    where g is nonlinearity, t is transform gate, and (1 - t) is carry gate.
    """
    output = input_
    for idx in xrange(layer_size):
        output = f(tf.nn.rnn_cell._linear(output, size, 0, scope='output_lin_%d' % idx))
        transform_gate = tf.sigmoid(
            tf.nn.rnn_cell._linear(input_, size, 0, scope='transform_lin_%d' % idx) + bias)
        carry_gate = 1. - transform_gate
        output = transform_gate * output + carry_gate * input_
    return output

