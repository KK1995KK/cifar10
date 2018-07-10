from keras import backend as K
from keras.engine.topology import Layer
import numpy as np

class WeightGraph2D(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(WeightGraph2D, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(WeightGraph2D, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        return K.dot(x, self.kernel)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

from keras import utils
from keras.datasets import fashion_mnist
from keras.models import Model
from keras.layers import *
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
num_classes = 10
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
y_train = utils.to_categorical(y_train, num_classes)
y_test = utils.to_categorical(y_test, num_classes)
print()