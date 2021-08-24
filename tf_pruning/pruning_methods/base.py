import tensorflow as tf

from abc import abstractmethod

from ..layers.base import PrunableLayer

class PruningMethod(tf.keras.layers.Wrapper):

    def __init__(
        self, 
        layer,
        pruning_schedule,
        **kwargs
    ):
        super(PruningMethod, self).__init__(layer, **kwargs)
        self.pruning_schedule = pruning_schedule

        if hasattr(self.layer, 'layers'):
            self.is_composite_layer = True
            self.children = [
                PruningMethod(child, pruning_schedule, **kwargs) 
                for child in self.layers if isinstance(child, PrunableLayer) 
            ]
        else:
            self.is_composite_layer = False
            self.chidren = []

    @abstractmethod
    def prune(self, **kwargs):
        pass

    def reset_step(self, epoch):
        self.pruning_schedule.current_step = epoch


class LayerwisePruningMethod(PruningMethod):

    @abstractmethod
    def prune_layer(self, **kwargs):
        pass

    def prune(self, **kwargs):
        if self.is_composite_layer:
            for child in self.chidren:
                child.prune()
        else:
            self.prune_layer(**kwargs)
            

# class ModelwisePruningMethod(PruningMethod):

#     @abstractmethod
#     def prune_layer(self, **kwargs):
#         pass

#     def prune(self, **kwargs):
#         if self.is_composite_layer:
#             for child in self.chidren:
#                 child.prune()
#         else:
#             self.prune_layer(**kwargs)