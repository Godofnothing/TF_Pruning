import tensorflow as tf
from tensorflow.python.keras.layers import Dense

from .base import PrunableLayer

class PrunableDense(Dense, PrunableLayer):

  def __init__(self, units, prune_bias=False, **kwargs):
    super(PrunableDense, self).__init__(units, **kwargs)
    self.prune_bias = prune_bias

  def build(self, input_shape):
    super(PrunableDense, self).build(input_shape)
    
    self.kernel_mask = self.add_weight(
        'kernel',
        shape=self.kernel.shape,
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint,
        dtype=self.dtype,
        trainable=True
    )

    if self.prune_bias:
        self.bias = self.add_weight(
          'bias',
          shape=[self.units,],
          initializer=self.bias_initializer,
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint,
          dtype=self.dtype,
          trainable=True
        )

  def call(self, inputs):
    if inputs.dtype.base_dtype != self._compute_dtype_object.base_dtype:
      inputs = tf.cast(inputs, dtype=self._compute_dtype_object)

    rank = inputs.shape.rank
    if rank == 2 or rank is None:
      # We use embedding_lookup_sparse as a more efficient matmul operation for
      # large sparse input tensors. The op will result in a sparse gradient, as
      # opposed to sparse_ops.sparse_tensor_dense_matmul which results in dense
      # gradients. This can lead to sigfinicant speedups, see b/171762937.
      if isinstance(inputs, tf.SparseTensor):
        # We need to fill empty rows, as the op assumes at least one id per row.
        inputs, _ = tf.sparse.fill_empty_rows(inputs, 0)
        # We need to do some munging of our input to use the embedding lookup as
        # a matrix multiply. We split our input matrix into separate ids and
        # weights tensors. The values of the ids tensor should be the column
        # indices of our input matrix and the values of the weights tensor
        # can continue to the actual matrix weights.
        # The column arrangement of ids and weights
        # will be summed over and does not matter. See the documentation for
        # sparse_ops.sparse_tensor_dense_matmul a more detailed explanation
        # of the inputs to both ops.
        ids = tf.SparseTensor(
            indices=inputs.indices,
            values=inputs.indices[:, 1],
            dense_shape=inputs.dense_shape)
        weights = inputs
        outputs = tf.nn.embedding_lookup_sparse(
            self.kernel, ids, weights, combiner='sum')
      else:
        outputs = tf.raw_ops.MatMul(a=inputs, b=self.kernel)
    # Broadcast kernel to inputs.
    else:
      # mask the kernel in order to zero gradients on the masked positions
      outputs = tf.tensordot(inputs, self.kernel_mask * self.kernel, [[rank - 1], [0]])
      # Reshape the output back to the original ndim of the input.
      if not tf.executing_eagerly():
        shape = inputs.shape.as_list()
        output_shape = shape[:-1] + [self.kernel.shape[-1]]
        outputs.set_shape(output_shape)

    if self.use_bias:
      outputs = tf.nn.bias_add(outputs, self.bias_mask * self.bias)

    if self.activation is not None:
      outputs = self.activation(outputs)
    return outputs
