import tensorflow as tf

def tf_num_elements(x : tf.Tensor)->int:
    return tf.size(x).numpy()

def tf_nonzero(x : tf.Tensor)->int:
    return tf.math.count_nonzero(x).numpy()

def tf_nonzero_fraction(x : tf.Tensor)->float:
    return tf_nonzero(x) / tf_num_elements(x)