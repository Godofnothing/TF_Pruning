from abc import abstractmethod

from tensorflow.keras.layers import Layer

class PrunableLayer(Layer):

    def __init__(self, **kwargs):
        super(PrunableLayer, self).__init__()

    @abstractmethod
    def get_weights_and_masks(self):
        pass

    @abstractmethod
    def build(self, input_shape):
        pass

    @abstractmethod
    def call(self, inputs, *args, **kwargs):
        pass
