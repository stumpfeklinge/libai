from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.keras import backend as K
from tensorflow.python.ops import nn
from tensorflow.python.keras.utils import control_flow_util
from tensorflow.python.keras.engine.input_spec import InputSpec


class Dropout(Layer):
    def __init__(self, rate, noise_shape=None, seed=None, **kwargs):
        super(Dropout, self).__init__(**kwargs)
        if isinstance(rate, (int, float)) and not 0 <= rate <= 1:
            raise ValueError(f'Invalid value {rate} received for '
                             f'`rate`, expected a value between 0 and 1.')
        self.rate = rate
        self.noise_shape = noise_shape
        self.seed = seed
        self.supports_masking = True

    def _get_noise_shape(self, inputs):
        # Subclasses of `Dropout` may implement `_get_noise_shape(self, inputs)`,
        # which will override `self.noise_shape`, and allows for custom noise
        # shapes with dynamically sized inputs.
        if self.noise_shape is None:
            return None

        concrete_inputs_shape = array_ops.shape(inputs)
        noise_shape = []
        for i, value in enumerate(self.noise_shape):
            noise_shape.append(concrete_inputs_shape[i] if value is None else value)
        return tensor_conversion.convert_to_tensor_v2_with_dispatch(noise_shape)

    def call(self, inputs, training=None):
        if training is None:
            training = K.learning_phase()

        def dropped_inputs():
            return nn.dropout(
                inputs,
                noise_shape=self._get_noise_shape(inputs),
                seed=self.seed,
                rate=self.rate)

        output = control_flow_util.smart_cond(training, dropped_inputs,
                                              lambda: array_ops.identity(inputs))
        return output

    def compute_output_shape(self, input_shape):
        return input_shape


class Dropout2D(Dropout):
    def __init__(self, rate, data_format=None, **kwargs):
        super(Dropout2D, self).__init__(rate, **kwargs)
        if data_format is None:
            data_format = K.image_data_format()
        if data_format not in {'channels_last', 'channels_first'}:
            raise ValueError('data_format must be in '
                             '{"channels_last", "channels_first"}')
        self.data_format = data_format
        self.input_spec = InputSpec(ndim=4)

    def _get_noise_shape(self, inputs):
        input_shape = array_ops.shape(inputs)
        if self.data_format == 'channels_first':
            return (input_shape[0], input_shape[1], 1, 1)
        elif self.data_format == 'channels_last':
            return (input_shape[0], 1, 1, input_shape[3])
