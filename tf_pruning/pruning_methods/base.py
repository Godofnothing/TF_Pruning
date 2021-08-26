import tensorflow as tf

from abc import abstractmethod

from tf_pruning.layers.base import PrunableLayer
from utils import has_sublayers

class PruningWrapper:

    def __init__(
        self, 
        layer : tf.keras.layers.Layer,
        pruning_schedule,
        **kwargs
    ):
      self.layer = layer
      self.pruning_schedule = pruning_schedule

      # create hierarchy of sublayers
      self.sublayers = []
      if has_sublayers(self.layer):
          for sublayer in self.layer.layers:
            if isinstance(sublayer, PruningWrapper):
              self.sublayers.append(sublayer)
            else:
              if has_sublayers(sublayer) or isinstance(sublayer, PrunableLayer):
                self.sublayers.append(self.__class__(sublayer, pruning_schedule, **kwargs))

    def set_epoch(self, epoch):
        self.pruning_schedule.current_epoch = epoch

    @abstractmethod
    def prune(self, **kwargs):
        pass

    @abstractmethod
    def prune_self(self, **kwargs):
        pass

    def __getattr__(self, name):
        return getattr(self.layer, name)

    def __repr__(self):
        return f"Wrapped_{self.layer.__class__.__name__}"


def make_wrapper_object(layer, pruning_schedule, *args, **kwargs):
    wrapper_obj = PruningWrapper(layer, pruning_schedule, *args, **kwargs)
    WrappedClass = type(f'Wrapped_{layer.__class__}', (layer.__class__, PruningWrapper, object), {})
    wrapper_obj.__class__ = WrappedClass
    return wrapper_obj


class LayerPruningMethod:
    '''
    This family of pruning methods performs pruning layerwise. 
    Pruning is performed in different layers independently.
    '''

    def prune_self(self, **kwargs)->None:
        pass

    def prune(self, **kwargs):
        for sublayer in self.sublayers:
            sublayer.prune(**kwargs)
        self.prune_self(**kwargs)

    def __new__(self, layer, pruning_schedule, *args, **kwargs):
        wrapper_obj = make_wrapper_object(layer, pruning_schedule, *args, **kwargs)

        # add methods specific for LayerPruningMethod
        wrapper_obj.prune_self = self.prune_self
        wrapper_obj.prune = self.prune

        return wrapper_obj
    

class ModelPruningMethod:
    '''
    This family of pruning methods performs pruning along the whole model - 
    collects all weights in the sublayers of the given layer and then performs the pruning.
    '''

    def prune_composite(self, prune_data):
        pass

    def prune_self(self, **kwargs)->list:
        return []

    def prune(self, is_top_layer=True, **kwargs):
        prune_data = []

        # collect data from sublayers
        for sublayer in self.sublayers:
            prune_data.extend(sublayer.prune(is_top_layer=False, **kwargs))
        # collect data from this layer
        prune_data.extend(self.prune_self(**kwargs))   

        if is_top_layer:
            self.prune_composite(prune_data, **kwargs)
        else:
            return prune_data

    def __new__(self, layer, pruning_schedule, *args, **kwargs):
        wrapper_obj = make_wrapper_object(layer, pruning_schedule, *args, **kwargs)

        # add methods specific for ModelPruningMethod
        wrapper_obj.prune_composite = self.prune_composite
        wrapper_obj.prune_self = self.prune_self
        wrapper_obj.prune = self.prune

        return wrapper_obj
            