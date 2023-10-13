import tensorflow as tf
from tensorflow.python.keras.engine.base_layer import Layer
class Pooling2D(Layer):

  def __init__(self, pool_function, pool_size, strides,
               padding='valid', data_format=None,
               name=None, **kwargs):
    super(Pooling2D, self).__init__(name=name, **kwargs)
    if data_format is None:
      data_format = tf.python.keras.backend.image_data_format()
    if strides is None:
      strides = pool_size
    self.pool_function = pool_function
    self.pool_size = tf.python.keras.utils.conv_utils.normalize_tuple(pool_size, 2, 'pool_size')
    self.strides = tf.python.keras.utils.conv_utils.normalize_tuple(strides, 2, 'strides')
    self.padding = tf.python.keras.utils.conv_utils.normalize_padding(padding)
    self.data_format = tf.python.keras.utils.conv_utils.normalize_data_format(data_format)
    self.input_spec = tf.python.keras.engine.input_spec.InputSpec(ndim=4)

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
        data_format=tf.python.keras.utils.conv_utils.convert_data_format(self.data_format, 4))
    return outputs

  def compute_output_shape(self, input_shape):
    input_shape = tf.python.framework.tensor_shape.TensorShape(input_shape).as_list()
    if self.data_format == 'channels_first':
      rows = input_shape[2]
      cols = input_shape[3]
    else:
      rows = input_shape[1]
      cols = input_shape[2]
    rows = tf.python.keras.utils.conv_utils.conv_output_length(rows, self.pool_size[0], self.padding,
                                         self.strides[0])
    cols = tf.python.keras.utils.conv_utils.conv_output_length(cols, self.pool_size[1], self.padding,
                                         self.strides[1])
    if self.data_format == 'channels_first':
      return tf.python.framework.tensor_shape.TensorShape(
          [input_shape[0], input_shape[1], rows, cols])
    else:
      return tf.python.framework.tensor_shape.TensorShape(
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


class MaxPooling2D(Pooling2D):
  def __init__(self,
               pool_size=(2, 2),
               strides=None,
               padding='valid',
               data_format=None,
               **kwargs):
    super(MaxPooling2D, self).__init__(
        tf.python.ops.nn.max_pool,
        pool_size=pool_size, strides=strides,
        padding=padding, data_format=data_format, **kwargs)