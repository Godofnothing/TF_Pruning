import tensorflow as tf

from tf_pruning.pruning_methods.base import LayerPruningMethod
from tf_pruning.layers.base import PrunableLayer
from utils import tf_num_elements

class L1PruningUnstructured(LayerPruningMethod):

    def prune_self(self, **kwargs):
        if not isinstance(self.layer, PrunableLayer):
            return
        
        for weight, mask in self.layer.get_weights_and_masks():
            sparsity = self.pruning_schedule.get_sparsity()
            n_nonzero = int(sparsity * tf_num_elements(mask))

            flattened_weight = tf.reshape(weight, shape=(-1,))
            pruning_scores = -tf.abs(flattened_weight)

            values, indices = tf.math.top_k(pruning_scores, k=n_nonzero, sorted=True)
            indices = tf.reshape(indices, shape=(-1, 1))

            new_mask = tf.scatter_nd(indices, tf.ones_like(values), shape=flattened_weight.shape)
            new_mask = tf.reshape(new_mask, shape=weight.shape)

            mask.assign(new_mask)
            weight.assign(mask * weight)
    