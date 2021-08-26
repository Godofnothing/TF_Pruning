from tensorflow.keras.callbacks import Callback

class PruningCallback(Callback):

    def on_train_begin(self, logs=None):
        if not hasattr(self.model, 'pruning_wrapped'):
            raise RuntimeWarning("Model is not wrapped by PruningMethod")

    def on_epoch_end(self, epoch, logs=None):
        self.model.set_epoch(epoch)
        self.model.prune()

