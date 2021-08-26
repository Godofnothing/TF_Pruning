import tensorflow as tf

def has_sublayers(x : tf.keras.layers.Layer)->bool:
    return hasattr(x, 'layers')

def get_hierarchy(layer : tf.keras.layers.Layer):
    if has_sublayers(layer, 'layers'):
        return [get_hierarchy(sublayer) for sublayer in layer.layers]
    else:
        return layer
        