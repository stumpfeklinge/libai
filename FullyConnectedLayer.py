import tensorflow as tf

class FullyConnectedLayer:
    def __init__(self, input_size, output_size):
        self.weights = tf.Variable(tf.random.normal([input_size, output_size]))
        self.biases = tf.Variable(tf.zeros([output_size]))

    def forward(self, inputs):
        return tf.matmul(inputs, self.weights) + self.biases

    def backward(self, inputs, grad):
        weights_grad = tf.matmul(tf.transpose(inputs), grad)
        biases_grad = tf.reduce_sum(grad, axis=0)
        inputs_grad = tf.matmul(grad, tf.transpose(self.weights))

        self.weights.assign_sub(weights_grad)
        self.biases.assign_sub(biases_grad)

        return inputs_grad