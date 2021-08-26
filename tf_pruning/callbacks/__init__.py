from tensorflow.keras.callbacks import Callback

from ..pruning_methods.base import PruningWrapper

class PruningManager(Callback):

    def on_train_begin(self, logs=None):
        if not isinstance(self.model, PruningWrapper):
            raise RuntimeWarning("There are no prunable layers in the model")

    def on_epoch_end(self, epoch, logs=None):
        self.model.set_epoch(epoch)
        self.model.prune()
