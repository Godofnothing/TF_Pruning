from tf_pruning import pruning_methods
import tensorflow as tf

from tf_pruning.pruning_methods.base import LayerPruningMethod
from tf_pruning.layers.base import PrunableLayer
from utils import tf_num_elements


class L1PruningUnstructured(LayerPruningMethod):

    def prune_self(self, **kwargs):
        # escape if layer is not prunable of it is not a pruning epoch
        if not isinstance(self, PrunableLayer) or not self.pruning_schedule.is_prune_epoch():
            return
        
        for weight, mask in self.get_weights_and_masks():
            sparsity = self.pruning_schedule.get_sparsity()
            n_nonzero = int(sparsity * tf_num_elements(mask))

            flattened_weight = tf.reshape(weight, shape=(-1,))
            pruning_scores = tf.abs(flattened_weight)

            values, indices = tf.math.top_k(pruning_scores, k=n_nonzero, sorted=True)
            indices = tf.reshape(indices, shape=(-1, 1))

            new_mask = tf.scatter_nd(indices, tf.ones_like(values), shape=flattened_weight.shape)
            new_mask = tf.reshape(new_mask, shape=weight.shape)

            mask.assign(new_mask)
            weight.assign(mask * weight)
    

class LnPruningStructured(LayerPruningMethod):

    def prune_self(self, **kwargs):
        # escape if layer is not prunable of it is not a pruning epoch
        if not isinstance(self, PrunableLayer) or not self.pruning_schedule.is_prune_epoch():
            return
        
        for weight, mask in self.get_weights_and_masks():
            sparsity = self.pruning_schedule.get_sparsity()
            n_nonzero = int(sparsity * tf_num_elements(tf.gather(mask, 0, axis=self.prune_axis)))

            pruning_scores = tf.norm(weight, axis=self.prune_axis, ord=self.prune_ord)
            pruning_scores = tf.reshape(pruning_scores, shape=(-1, ))
            values, indices = tf.math.top_k(pruning_scores, k=n_nonzero, sorted=True)
            indices = tf.reshape(indices, shape=(-1, 1))

            new_mask = tf.scatter_nd(indices, tf.ones_like(values), shape=pruning_scores.shape)
            # expand along the axis on which the pruning is performed
            new_mask = tf.ones_like(mask) * tf.expand_dims(new_mask, axis=self.prune_axis)

            mask.assign(new_mask)
            weight.assign(mask * weight)

    def __new__(self, layer, pruning_schedule, axis=0, ord=2, *args, **kwargs):
      layer = super(L1PruningUnstructured, self).__new__(self, layer, pruning_schedule, *args, **kwargs)
      layer.prune_axis = axis
      layer.prune_ord = ord
      return layer
      