from __future__ import print_function
import math
import time
import numpy as np
import tensorflow as tf

def create_padding_dimension(n):
    # Halve the padded dimension and round up the first result, and round down the second to create a correct
    # padding dimension.
    return [int(math.ceil(n / 2.0)), int(math.floor(n / 2.0))]

def zero_pad(input, shape):
    #print "input shape %s" % np.array(input.get_shape().as_list())
    #print "target shape %s" % np.array(shape)
    input_shape = input.get_shape().as_list()
    shape_disparity = np.array(shape) - np.array(input_shape)
    for i in range(len(shape)):
        if shape[i] == -1:
            shape_disparity[i] = 0
    padding = list(map(create_padding_dimension, shape_disparity))
    #print "Zero pad: padding %s" % padding
    return tf.pad(input, padding)

def depth_concat(values):
    # This method assumes that all values will be in the shape of (batch, x, y, filters), 
    # where batch is equal for all tensors and the rest of the dimensions may vary.
    
    # The output of this method will have a shape (batch, output_x, output_y, total_feature_maps),
    # where output_x and output_y is the largest dimensions out of all of the input layers, 
    # total_feature_maps is the number of feature maps in all concatenated layers.

    # In neural networks, depth can refer to the number of layers in a model, but it can also refer to the number of channels in an 'activation volume'.
    # http://cs231n.github.io/convolutional-networks/#conv states:
    # 'In particular, unlike a regular Neural Network, the layers of a ConvNet have neurons arranged in 3 dimensions: width, height, depth. 
    # (Note that the word depth here refers to the third dimension of an activation volume, not to the depth of a full Neural Network, which 
    # can refer to the total number of layers in a network.)'.
    # 
    # so in this case the depth is the channels index, the fourth dimension.
    batch_index = 0
    x_index = 1
    y_index = 2
    depth_index = 3 
    max_x = max(map(lambda a: a.get_shape().as_list()[x_index], values))
    max_y = max(map(lambda a: a.get_shape().as_list()[y_index], values))
    #print max_x, max_y
    batch_size = values[0].get_shape().as_list()[batch_index] # Assume all values have this dimension value
    depth_size = sum(map(lambda a: a.get_shape().as_list()[depth_index], values))
    shape = [batch_size, max_x, max_y, -1]
    padded_values = list(map(lambda a: zero_pad(a, shape), values))
    #print(padded_values)
    return tf.concat(depth_index, padded_values)

def get_filter_output_size(input_size, filter_size, filter_stride):
        return ((input_size - filter_size) // filter_stride) + 1

def get_l2_loss(l2_lambda, layer_weights):
    return l2_lambda * sum(map(tf.nn.l2_loss, layer_weights))
