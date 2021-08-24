import tensorflow as tf
from tensorflow.keras.callbacks import Callback

from ..pruning_methods.base import PruningMethod


def get_prunable_layers(model):
    return [layer for layer in model.layers if isinstance(layer, PruningMethod)]

class PruningManager(Callback):

    def __init__(self):
        super(PruningManager, self).__init__()

    def on_train_begin(self, logs=None):
        self.prunable_layers = get_prunable_layers(self.model)
        if not self.prunable_layers:
            raise RuntimeWarning("There are no prunable layers in the model")
        
        for prunable_layer in self.prunable_layers:
            prunable_layer.reset_step(0)

    def on_epoch_end(self, epoch, logs=None):
        for prunable_layer in self.prunable_layers:
            prunable_layer.prune()
            prunable_layer.reset_step(epoch)
