import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.python.keras.layers.convolutional import Conv
from tensorflow.python.keras.utils import conv_utils

from .base import PrunableLayer


class PrunableConv(Conv, PrunableLayer):

    def __init__(self, rank, filters, kernel_size, prune_bias=False, **kwargs):
        super(PrunableConv, self).__init__(
            rank=rank,
            filters=filters, 
            kernel_size=kernel_size,
            **kwargs
        )

        self.prune_bias = prune_bias

    def build(self, input_shape):
        super(PrunableConv, self).build(input_shape)

        self.kernel_mask = self.add_weight(
            name='kernel_mask',
            shape=self.kernel.shape,
            initializer='ones',
            trainable=False,
            dtype=self.kernel.dtype
        )

        if self.prune_bias:
            self.bias_mask = self.add_weight(
                name='bias_mask',
                shape=self.bias.shape,
                initializer='ones',
                trainable=False,
                dtype=self.bias.dtype
            )

    def get_weights_and_masks(self):
      weights_and_masks = [(self.kernel, self.kernel_mask)]
      if self.prune_bias:
        weights_and_masks += [(self.bias, self.bias_mask)]
      return weights_and_masks

      
    def call(self, inputs):
      input_shape = inputs.shape

      if self._is_causal:  # Apply causal padding to inputs for Conv1D.
        inputs = tf.pad(inputs, self._compute_causal_padding(inputs))

      outputs = self._convolution_op(inputs, self.kernel * self.kernel_mask)

      if self.use_bias:
        output_rank = outputs.shape.rank

        if self.prune_bias:
          bias = self.bias * self.bias_mask
        else:
          bias = self.bias

        if self.rank == 1 and self._channels_first:
          # nn.bias_add does not accept a 1D input tensor.
          bias = tf.reshape(bias, (1, self.filters, 1))
          outputs += bias
        else:
          # Handle multiple batch dimensions.
          if output_rank is not None and output_rank > 2 + self.rank:

            def _apply_fn(o):
              return tf.nn.bias_add(o, bias, data_format=self._tf_data_format)

            outputs = conv_utils.squeeze_batch_dims(
                outputs, _apply_fn, inner_rank=self.rank + 1)
          else:
            outputs = tf.nn.bias_add(
                outputs, bias, data_format=self._tf_data_format)

      if not tf.executing_eagerly():
        # Infer the static output shape:
        out_shape = self.compute_output_shape(input_shape)
        outputs.set_shape(out_shape)

      if self.activation is not None:
        return self.activation(outputs)
      return outputs


class PrunableConv1D(PrunableConv):

    def __init__(self, filters, kernel_size, prune_bias=False, **kwargs):
        super(PrunableConv1D, self).__init__(
            rank=1,
            filters=filters, 
            kernel_size=kernel_size,
            prune_bias=prune_bias,
            **kwargs
        )


class PrunableConv2D(PrunableConv):

    def __init__(self, filters, kernel_size, prune_bias=False, **kwargs):
        super(PrunableConv2D, self).__init__(
            rank=2,
            filters=filters, 
            kernel_size=kernel_size,
            prune_bias=prune_bias,
            **kwargs
        )


class PrunableConv3D(PrunableConv):

    def __init__(self, filters, kernel_size, prune_bias=False, **kwargs):
        super(PrunableConv2D, self).__init__(
            rank=3,
            filters=filters, 
            kernel_size=kernel_size,
            prune_bias=prune_bias,
            **kwargs
        )
