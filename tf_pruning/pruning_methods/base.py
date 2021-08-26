from abc import abstractmethod
from types import MethodType

from tf_pruning.layers.base import PrunableLayer
from utils import has_sublayers

class PruningMethod:
    '''
    Base class for all pruning methods
    '''

    def _init_sublayers(self, layer, **kwargs):
      # init sublayers
      layer.sublayers = []
      if has_sublayers(layer):
          for sublayer in layer.layers:
            if hasattr(sublayer, 'pruning_wrapped'):
              layer.sublayers.append(sublayer)
            else:
              if has_sublayers(sublayer) or isinstance(sublayer, PrunableLayer):
                layer.sublayers.append(self.__new__(self, sublayer, layer.pruning_schedule, **kwargs))

    def set_epoch(self, epoch):
        self.pruning_schedule.current_epoch = epoch

    @abstractmethod
    def prune(self, **kwargs):
      pass

    @abstractmethod
    def prune_self(self, **kwargs):
      pass

    def __new__(self, layer, pruning_schedule, *args, **kwargs):
      layer.pruning_schedule = pruning_schedule
      layer.set_epoch = MethodType(self.set_epoch, layer)
      layer.prune = MethodType(self.prune, layer)
      layer.prune_self = MethodType(self.prune_self, layer)

      self._init_sublayers(self, layer, **kwargs)

      layer.pruning_wrapped = True

      return layer

class LayerPruningMethod(PruningMethod):
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
    

class ModelPruningMethod(PruningMethod):
    '''
    This family of pruning methods performs pruning along the whole model - 
    collects all weights in the sublayers of the given layer and then performs the pruning.
    '''
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

    def prune_composite(self, prune_data):
      pass

    def __new__(self, layer, pruning_schedule, *args, **kwargs):
      layer = super(ModelPruningMethod, self).__new__(self, layer, pruning_schedule, *args, **kwargs)
      layer.prune_composite = MethodType(self.prune_composite, layer)
      return layer
