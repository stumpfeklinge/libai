from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import tensor_shape
import functools
import operator
import numpy as np
class Flatten2D(Layer):


  def __init__(self, data_format=None, **kwargs):
    super(Flatten2D, self).__init__(**kwargs)
    self.data_format = conv_utils.normalize_data_format(data_format)
    self.input_spec = InputSpec(min_ndim=1)
    self._channels_first = self.data_format == 'channels_first'

  def call(self, inputs):
    if self._channels_first:
      rank = inputs.shape.rank
      if rank and rank > 1:
        # Switch to channels-last format.
        permutation = [0]
        permutation.extend(range(2, rank))
        permutation.append(1)
        inputs = array_ops.transpose(inputs, perm=permutation)

    if context.executing_eagerly():
      # Full static shape is guaranteed to be available.
      # Performance: Using `constant_op` is much faster than passing a list.
      flattened_shape = constant_op.constant([inputs.shape[0], -1])
      return array_ops.reshape(inputs, flattened_shape)
    else:
      input_shape = inputs.shape
      rank = input_shape.rank
      if rank == 1:
        return array_ops.expand_dims_v2(inputs, axis=1)
      else:
        batch_dim = tensor_shape.dimension_value(input_shape[0])
        non_batch_dims = input_shape[1:]
        # Reshape in a way that preserves as much shape info as possible.
        if non_batch_dims.is_fully_defined():
          last_dim = int(functools.reduce(operator.mul, non_batch_dims))
          flattened_shape = constant_op.constant([-1, last_dim])
        elif batch_dim is not None:
          flattened_shape = constant_op.constant([int(batch_dim), -1])
        else:
          flattened_shape = [array_ops.shape_v2(inputs)[0], -1]
        return array_ops.reshape(inputs, flattened_shape)

  def compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    if not input_shape:
      output_shape = tensor_shape.TensorShape([1])
    else:
      output_shape = [input_shape[0]]
    if np.all(input_shape[1:]):
      output_shape += [np.prod(input_shape[1:], dtype=int)]
    else:
      output_shape += [None]
    return tensor_shape.TensorShape(output_shape)

  def get_config(self):
    config = super(Flatten2D, self).get_config()
    config.update({'data_format': self.data_format})
    return config
