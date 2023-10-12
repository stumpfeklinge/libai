import tensorflow as tf

class Conv2DLayer(tf.keras.layers.Layer):
    def __init__(self, num_filters, kernel_size):
        super(Conv2DLayer, self).__init__()
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.conv_layer = tf.keras.layers.Conv2D(num_filters, kernel_size)

    def call(self, inputs):
        output = self.conv_layer(inputs)
        return output

# Пример использования
'''
input_shape = (32, 32, 3)
num_filters = 64
kernel_size = (3, 3)
input_data = tf.random.normal((1,) + input_shape)

conv_layer = Conv2DLayer(num_filters, kernel_size)
output_data = conv_layer(input_data)

print(f"Input shape: {input_data.shape}")
print(f"Output shape: {output_data.shape}")
'''
