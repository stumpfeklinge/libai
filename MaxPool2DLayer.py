from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.ops import nn
from tensorflow import keras
from tensorflow.python.keras import backend
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.utils import conv_utils

class Pooling2DLayer(Layer):

  def __init__(self, pool_function, pool_size, strides,
               padding='valid', data_format=None,
               name=None, **kwargs):
    super(Pooling2DLayer, self).__init__(name=name, **kwargs)
    if data_format is None:
      data_format = backend.image_data_format()
    if strides is None:
      strides = pool_size
    self.pool_function = pool_function
    self.pool_size = conv_utils.normalize_tuple(pool_size, 2, 'pool_size')
    self.strides = conv_utils.normalize_tuple(strides, 2, 'strides')
    self.padding = conv_utils.normalize_padding(padding)
    self.data_format = conv_utils.normalize_data_format(data_format)
    self.input_spec = InputSpec(ndim=4)

  def call(self, inputs):
    if self.data_format == 'channels_last':
      pool_shape = (1,) + self.pool_size + (1,)
      strides = (1,) + self.strides + (1,)
    else:
      pool_shape = (1, 1) + self.pool_size
      strides = (1, 1) + self.strides
    outputs = self.pool_function(
        inputs,
        ksize=pool_shape,
        strides=strides,
        padding=self.padding.upper(),
        data_format=conv_utils.convert_data_format(self.data_format, 4))
    return outputs

  def compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    if self.data_format == 'channels_first':
      rows = input_shape[2]
      cols = input_shape[3]
    else:
      rows = input_shape[1]
      cols = input_shape[2]
    rows = conv_utils.conv_output_length(rows, self.pool_size[0], self.padding,
                                         self.strides[0])
    cols = conv_utils.conv_output_length(cols, self.pool_size[1], self.padding,
                                         self.strides[1])
    if self.data_format == 'channels_first':
      return tensor_shape.TensorShape(
          [input_shape[0], input_shape[1], rows, cols])
    else:
      return tensor_shape.TensorShape(
          [input_shape[0], rows, cols, input_shape[3]])

  def get_config(self):
    config = {
        'pool_size': self.pool_size,
        'padding': self.padding,
        'strides': self.strides,
        'data_format': self.data_format
    }
    base_config = super(Pooling2D, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class MaxPool2DLayer(Pooling2D):

  def __init__(self,
               pool_size=(2, 2),
               strides=None,
               padding='valid',
               data_format=None,
               **kwargs):
    super(MaxPool2DLayer, self).__init__(
        nn.max_pool,
        pool_size=pool_size, strides=strides,
        padding=padding, data_format=data_format, **kwargs)
