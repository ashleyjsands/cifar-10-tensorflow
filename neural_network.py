import math

import numpy as np
import tensorflow as tf

def create_padding_dimension(n):
    # Halve the padded dimension and round up the first result, and round down the second to create a correct
    # padding dimension.
    return [int(math.ceil(n / 2.0)), int(math.floor(n / 2.0))]

def zero_pad(input, shape):
    print "input shape %s" % np.array(input.get_shape().as_list())
    print "target shape %s" % np.array(shape)
    shape_disparity = np.array(shape) - np.array(input.get_shape().as_list())
    padding = map(create_padding_dimension, shape_disparity)
    print "Zero pad: padding %s" % padding
    return tf.pad(input, padding)

def depth_concat(values):
    # This method assumes that all values will be in the shape of (batch, x, y, filters), 
    # where batch is equal for all tensors and the rest of the dimensions may vary.
    
    # The output of this method will have a shape (batch, output_x, output_y, total_feature_maps),
    # where output_x and output_y is the largest dimensions out of all of the input layers, 
    # yotal_feature_maps is the number of feature maps in all concatenated layers.
    depth_index = 0
    x_index = 1
    y_index = 2
    rgb_index = 3
    max_x = max(map(lambda a: a.get_shape().as_list()[x_index], values))
    max_y = max(map(lambda a: a.get_shape().as_list()[y_index], values))
    print max_x, max_y
    depth_size = values[0].get_shape().as_list()[depth_index] # Assume all values have this dimension value
    rgb_size = values[0].get_shape().as_list()[rgb_index] # Assume all values have this dimension value
    shape = [depth_size, max_x, max_y, rgb_size]
    padded_values = map(lambda a: zero_pad(a, shape), values)
    return tf.concat(depth_index, padded_values)
