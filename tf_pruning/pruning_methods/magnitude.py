import tensorflow as tf

from .base import LayerwisePruningMethod


class L1PruningUnstructured(LayerwisePruningMethod):

    def prune_layer(self, **kwargs):
        for weight, mask in self.layer.get_weights_and_masks():
            old_sparsity = tf.math.count_nonzero(mask)
            new_sparsity = self.pruning_schedule.sparsity()

            n_to_prune = (old_sparsity - new_sparsity) * weight.shape.num_elements()

            if n_to_prune <= 0:
                return 

            flattened_weight = tf.reshape(weight, shape=(-1,))
            pruning_scores = -tf.abs(flattened_weight)

            values, indices = tf.math.top_k(pruning_scores, k=n_to_prune, sorted=True)
            indices = tf.reshape(indices, shape=(-1, 1))

            new_mask = tf.scatter_nd(indices, tf.ones_like(values), shape=flattened_weight.shape)
            new_mask = tf.reshape(new_mask, shape=weight.shape)

            mask.assign(new_mask)
            weight.assign(mask * weight)
    