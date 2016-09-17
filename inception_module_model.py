import functools
import tensorflow as tf

from data import image_size, num_channels, num_labels
from model import Model
from neural_network import get_filter_output_size, depth_concat, get_l2_loss


print_messages = False

class Module():

    def __init__(self, one_by_one_conv_weights_to_three_by_three, one_by_one_conv_biases_to_three_by_three, one_by_one_conv_weights_to_five_by_five, one_by_one_conv_biases_to_five_by_five,
            one_by_one_conv_weights_to_depthconcat, one_by_one_conv_biases_to_depthconcat, one_by_one_conv_weights_from_max_pool, one_by_one_conv_biases_from_max_pool, three_by_three_conv_weights,
            three_by_three_conv_biases, five_by_five_conv_weights, five_by_five_conv_biases):
        self.one_by_one_conv_weights_to_three_by_three = one_by_one_conv_weights_to_three_by_three
        self.one_by_one_conv_biases_to_three_by_three = one_by_one_conv_biases_to_three_by_three
        self.one_by_one_conv_weights_to_five_by_five = one_by_one_conv_weights_to_five_by_five
        self.one_by_one_conv_biases_to_five_by_five = one_by_one_conv_biases_to_five_by_five
        self.one_by_one_conv_weights_to_depthconcat = one_by_one_conv_weights_to_depthconcat
        self.one_by_one_conv_biases_to_depthconcat = one_by_one_conv_biases_to_depthconcat
        self.one_by_one_conv_weights_from_max_pool = one_by_one_conv_weights_from_max_pool
        self.one_by_one_conv_biases_from_max_pool = one_by_one_conv_biases_from_max_pool
        self.three_by_three_conv_weights = three_by_three_conv_weights
        self.three_by_three_conv_biases = three_by_three_conv_biases
        self.five_by_five_conv_weights = five_by_five_conv_weights
        self.five_by_five_conv_biases = five_by_five_conv_biases

    def get_weights(self):
        return [self.one_by_one_conv_weights_to_three_by_three, self.one_by_one_conv_weights_to_five_by_five, self.one_by_one_conv_weights_to_depthconcat,
                self.one_by_one_conv_weights_from_max_pool, self.three_by_three_conv_weights, self.five_by_five_conv_weights]

def create_inception_module(in_channels, out_channels, input_spatial_size, initialised_weights_stddev):
    """
    Inception module:
    input layer
                1x1+1 conv, 1x1+1 conv, 3x3+1 maxpool
    1x1+1 conv, 3x3+1 conv, 5x5+1 conv, 1x1+1 conv
    DepthConcat
    AveragePool
    """
    patch_size = 1
    one_by_one_conv_weights_to_three_by_three = tf.Variable(tf.truncated_normal(
        [patch_size, patch_size, in_channels, out_channels], stddev=initialised_weights_stddev), name='1x1w_to_3x3')
    one_by_one_conv_biases_to_three_by_three = tf.Variable(tf.constant(initialised_weights_stddev * 10, shape=[out_channels]), name='1x1b_to_3x3')
    
    one_by_one_conv_weights_to_five_by_five = tf.Variable(tf.truncated_normal(
        [patch_size, patch_size, in_channels, out_channels], stddev=initialised_weights_stddev), name='1x1w_to_5x5')
    one_by_one_conv_biases_to_five_by_five = tf.Variable(tf.constant(initialised_weights_stddev * 10, shape=[out_channels]), name='1x1b_to_5x5')
    
    one_by_one_conv_weights_to_depthconcat = tf.Variable(tf.truncated_normal(
        [patch_size, patch_size, in_channels, out_channels], stddev=initialised_weights_stddev), name='1x1w_to_depth_concat')
    one_by_one_conv_biases_to_depthconcat = tf.Variable(tf.constant(initialised_weights_stddev * 10, shape=[out_channels]), name='1x1b_to_depth_concat')
    
    #number_of_max_pools = 1
    #three_by_three_max_pool_output_size = get_filter_output_size(image_size, number_of_max_pools)
    max_pool_size = 3
    max_pool_stride = 1
    three_by_three_max_pool_output_size = get_filter_output_size(input_spatial_size, max_pool_size, max_pool_stride)
    if print_messages: print("three_by_three_max_pool_output_size: %s" % three_by_three_max_pool_output_size)
    one_by_one_conv_weights_from_max_pool = tf.Variable(tf.truncated_normal(
        [three_by_three_max_pool_output_size, three_by_three_max_pool_output_size, in_channels, out_channels], stddev=initialised_weights_stddev),
                                                        name='1x1w_from_maxpool')
    one_by_one_conv_biases_from_max_pool = tf.Variable(tf.constant(initialised_weights_stddev * 10, shape=[out_channels]), name='1x1b_from_maxpool')
    
    patch_size = 3
    three_by_three_conv_weights = tf.Variable(tf.truncated_normal(
        [patch_size, patch_size, out_channels, out_channels], stddev=initialised_weights_stddev), name='3x3w')
    three_by_three_conv_biases = tf.Variable(tf.constant(initialised_weights_stddev * 10, shape=[out_channels]), name='3x3b')
    
    patch_size = 5
    five_by_five_conv_weights = tf.Variable(tf.truncated_normal(
        [patch_size, patch_size, out_channels, out_channels], stddev=initialised_weights_stddev), name='5x5w')
    five_by_five_conv_biases = tf.Variable(tf.constant(initialised_weights_stddev * 10, shape=[out_channels]), name='5x5b')

    # The 3x3 maxpooling layer, DepthConcat layer, and averagepooling layer don't need any variables.
    return Module(one_by_one_conv_weights_to_three_by_three, one_by_one_conv_biases_to_three_by_three, one_by_one_conv_weights_to_five_by_five, one_by_one_conv_biases_to_five_by_five,
            one_by_one_conv_weights_to_depthconcat, one_by_one_conv_biases_to_depthconcat, one_by_one_conv_weights_from_max_pool, one_by_one_conv_biases_from_max_pool, three_by_three_conv_weights,
            three_by_three_conv_biases, five_by_five_conv_weights, five_by_five_conv_biases)

def create_inception_module_graph(module, input_tensor):
    # Adjacent layer 1
    if print_messages: print("input_tensor shape:", input_tensor.get_shape().as_list())
    if print_messages: print("one_by_one_conv_weights_to_three_by_three shape:", module.one_by_one_conv_weights_to_three_by_three.get_shape().as_list())
    conv = tf.nn.conv2d(input_tensor, module.one_by_one_conv_weights_to_three_by_three, [1, 1, 1, 1], padding='SAME')
    one_by_one_conv_weights_to_three_by_three_output = tf.nn.relu(conv + module.one_by_one_conv_biases_to_three_by_three)
    if print_messages: print("one_by_one_conv_weights_to_three_by_three_output shape:", one_by_one_conv_weights_to_three_by_three_output.get_shape().as_list())

    conv = tf.nn.conv2d(input_tensor, module.one_by_one_conv_weights_to_five_by_five, [1, 1, 1, 1], padding='SAME')
    one_by_one_conv_weights_to_five_by_five_output = tf.nn.relu(conv + module.one_by_one_conv_biases_to_five_by_five)
    shape = one_by_one_conv_weights_to_five_by_five_output.get_shape().as_list()
    if print_messages: print("one_by_one_conv_weights_to_five_by_five_output shape: %s" % shape)

    max_pool_output = tf.nn.max_pool(input_tensor, [1, 3, 3, 1], [1, 1, 1, 1], padding='SAME')
    shape = max_pool_output.get_shape().as_list()
    if print_messages: print("max_pool_output shape: %s" % shape)

    # Adjacent layer 2
    conv = tf.nn.conv2d(input_tensor, module.one_by_one_conv_weights_to_depthconcat, [1, 1, 1, 1], padding='SAME')
    one_by_one_output_to_depthconcat = tf.nn.relu(conv + module.one_by_one_conv_biases_to_depthconcat)
    shape = one_by_one_output_to_depthconcat.get_shape().as_list()
    if print_messages: print("one_by_one_output_to_depthconcat shape: %s" % shape)

    conv = tf.nn.conv2d(one_by_one_conv_weights_to_three_by_three_output, module.three_by_three_conv_weights, [1, 1, 1, 1], padding='SAME')
    three_by_three_output = tf.nn.relu(conv + module.three_by_three_conv_biases)
    shape = three_by_three_output.get_shape().as_list()
    if print_messages: print("three_by_three_output shape: %s" % shape)

    conv = tf.nn.conv2d(one_by_one_conv_weights_to_five_by_five_output, module.five_by_five_conv_weights, [1, 1, 1, 1], padding='SAME')
    five_by_five_output = tf.nn.relu(conv + module.five_by_five_conv_biases)
    shape = five_by_five_output.get_shape().as_list()
    if print_messages: print("five_by_five_output shape: %s" % shape)

    if print_messages: print("one_by_one_conv_weights_from_max_pool:", module.one_by_one_conv_weights_from_max_pool.get_shape().as_list())
    conv = tf.nn.conv2d(max_pool_output, module.one_by_one_conv_weights_from_max_pool, [1, 1, 1, 1], padding='SAME')
    one_by_one_conv_weights_from_max_pool_output = tf.nn.relu(conv + module.one_by_one_conv_biases_from_max_pool)
    shape = one_by_one_conv_weights_from_max_pool_output.get_shape().as_list()
    if print_messages: print("one_by_one_conv_weights_from_max_pool_output shape: %s" % shape)
    
    depth_concat_output = depth_concat([one_by_one_output_to_depthconcat, three_by_three_output, five_by_five_output, one_by_one_conv_weights_from_max_pool_output])
    shape = depth_concat_output.get_shape().as_list()
    if print_messages: print("depth_concat_output shape: %s" % shape)
    return depth_concat_output

def create_inception_module_model(learning_rate = 0.05, initialised_weights_stddev = 0.1, pre_layer_feature_maps = 64, module_feature_maps=[128], batch_size = 32, eval_batch_size = 1000, l2_lambda = 0.1, decay_steps = 10000, decay_rate = 0.96, add_pre_layer_maxpool = True):
    graph = tf.Graph()
    with graph.as_default():

        # Input data.
        tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))
        tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
        eval_dataset = tf.placeholder(tf.float32, shape=(eval_batch_size, image_size, image_size, num_channels))
        dropout_keep_probability = tf.placeholder(tf.float32)
        """
        In the inception module, we have the following layers:
        Pre layers:
        7x7 conv + 2(S)
	maxpool 3x3 + 2(s)
	local response normalisation

        #1x1 conv + 1(V)
	#maxpool 3x3 + 1(s)
	#local response normalisation
        
        Inception modules: 1
        
        Post layers:
        AveragePool
        
        Fully Connected Output layer
        """
        
        # Pre layers
        post_layer_output_feature_maps = pre_layer_feature_maps
        patch_size = 7
        stride = 2
        seven_by_seven_conv_pre_layer_weights = tf.Variable(tf.truncated_normal(
            [patch_size, patch_size, num_channels, post_layer_output_feature_maps], stddev=initialised_weights_stddev), name='7x7_pre_w')
        seven_by_seven_conv_pre_layer_biases = tf.Variable(tf.constant(initialised_weights_stddev * 10, shape=[post_layer_output_feature_maps]), name='7x7_pre_b')
        if print_messages: print("image_size:", image_size)
        seven_by_seven_conv_pre_layer_output_size = get_filter_output_size(image_size, patch_size, stride)
        if print_messages: print("seven_by_seven_conv_pre_layer_output_size:", seven_by_seven_conv_pre_layer_output_size)

        three_by_three_maxpool_pre_layer_output_size = get_filter_output_size(seven_by_seven_conv_pre_layer_output_size, patch_size, stride)
        #pre_layer_output_spatial_size
        if print_messages: print("three_by_three_maxpool_pre_layer_output_size:", three_by_three_maxpool_pre_layer_output_size)
        
        # Module layers
        modules = []
        in_channels = post_layer_output_feature_maps 
        in_spatial_size = three_by_three_maxpool_pre_layer_output_size if add_pre_layer_maxpool else seven_by_seven_conv_pre_layer_output_size
        number_of_reducing_layers = 2
        stride = 1
        number_of_adjacent_layers_in_inception_module = 4
        for out_channels in module_feature_maps:
            if print_messages: print("in_spatial_size:", in_spatial_size)
            module = create_inception_module(in_channels, out_channels, in_spatial_size, initialised_weights_stddev)
            modules.append(module)
            in_channels = number_of_adjacent_layers_in_inception_module * out_channels
            in_spatial_size = get_filter_output_size(in_spatial_size, number_of_reducing_layers, stride)

        # Now a fully connected layer
        depth_concat_depth = in_channels #module_feature_maps[-1] * number_of_adjacent_layers_in_inception_module
        # I expect avg_pool_ouput to have a shape of (batch_size, 1, 1, depth_concat_depth)
        # WARNING: I may have gotten the fc_weights tensor size wrong.
        #fc_layer_one_neurons = 25
        #fc_layer_one_weights = tf.Variable(tf.truncated_normal(
        #    [depth_concat_depth, fc_layer_one_neurons], stddev=initialised_weights_stddev * 10))
        #fc_layer_one_biases = tf.Variable(tf.constant(initialised_weights_stddev * 10, shape=[fc_layer_one_neurons]))
        
        output_weights = tf.Variable(tf.truncated_normal(
            [depth_concat_depth, num_labels], stddev=initialised_weights_stddev))
        output_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))
        
        # Model.
        def create_model_graph(data, add_dropout = False):
            shape = data.get_shape().as_list()
            if print_messages: print("data shape: %s" % shape)
            
            # Pre layers
            # 7x7_pre
            stride = 2
            conv = tf.nn.conv2d(data, seven_by_seven_conv_pre_layer_weights, [1, stride, stride, 1], padding='SAME')
            seven_by_seven_conv_pre_layer_output = tf.nn.relu(conv + seven_by_seven_conv_pre_layer_biases)
            if print_messages: print("one_by_one_conv_weights_to_three_by_three_output shape: %s" % seven_by_seven_conv_pre_layer_output.get_shape().as_list())
            
            lrn_input = None
            if add_pre_layer_maxpool:
                # 3x3_mp
                patch_size = 3
                stride = 2
                max_pool_output = tf.nn.max_pool(seven_by_seven_conv_pre_layer_output, [1, patch_size, patch_size, 1], [1, stride, stride, 1], padding='SAME')
                lrn_input = max_pool_output
            else:
                lrn_input = seven_by_seven_conv_pre_layer_output
            
            # https://www.tensorflow.org/versions/r0.10/api_docs/python/nn.html#local_response_normalization
            # TODO: tune the following hyperparameters: depth_radius, bias, alpha, beta
            lrn_output = tf.nn.local_response_normalization(lrn_input)
            
            pre_layers_output = lrn_output
            
            # Inception module(s)
            input_tensor = pre_layers_output
            modules_output = None
            for module in modules:
                # The last module will set modules_output for use below.
                if print_messages: print("input_tensor shape:", input_tensor.get_shape().as_list())
                modules_output = create_inception_module_graph(module, input_tensor)  
                input_tensor = modules_output

            # The patch size of the avg_pool must match the patch_size of the depth_concat_output
            # I assume that the padding must be VALID based on Google's white paper: http://arxiv.org/pdf/1409.4842v1.pdf
            depth_concat_output_image_size_index = 1
            largest_patch_size = modules_output.get_shape().as_list()[depth_concat_output_image_size_index]
            if print_messages: print("largest_patch_size:", largest_patch_size)
            avg_pool_output = tf.nn.avg_pool(modules_output, [1, largest_patch_size, largest_patch_size, 1], [1, 1, 1, 1], padding='VALID', name=None)
            shape = avg_pool_output.get_shape().as_list()
            if print_messages: print("avg_pool_output shape:", shape)

            # Flatten the average_pool_output from 4 dimensions down to 2.
            batch_index = 0
            reshape_tensor = tf.reshape(avg_pool_output, (data.get_shape().as_list()[batch_index], 1 * 1 * depth_concat_depth))
            if print_messages: print("reshape_tensor shape: %s" % reshape_tensor.get_shape().as_list())
            
            # Post layers
            # N/A
            
            #fc_layer_one_output = tf.nn.relu(tf.matmul(reshape_tensor, fc_layer_one_weights) + fc_layer_one_biases)
            #if print_messages: print("fc_layer_one_output shape: %s" % fc_layer_one_output.get_shape().as_list())
            # TODO: add dropout.
            #if add_dropout:
            #    hidden = tf.nn.dropout(hidden, dropout_keep_probability)
            return tf.matmul(reshape_tensor, output_weights) + output_biases

        # Training computation.
        logits = create_model_graph(tf_train_dataset, add_dropout = True)
        module_weights = functools.reduce(lambda a, b: a + b, map(lambda m: m.get_weights(), modules))
        layer_weights = [ seven_by_seven_conv_pre_layer_weights ] + module_weights + [ output_weights ]
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels) + get_l2_loss(l2_lambda, layer_weights))

        # Optimizer.
        global_step = tf.Variable(0, name="global_step", trainable=False)
        decayed_learning_rate = tf.train.exponential_decay(learning_rate, global_step, decay_steps, decay_rate)
        optimizer = tf.train.GradientDescentOptimizer(decayed_learning_rate).minimize(loss, global_step=global_step)

        # Predictions for the training, validation, and test data.
        train_prediction = tf.nn.softmax(logits)
        eval_prediction = tf.nn.softmax(create_model_graph(eval_dataset))
        
        saver = tf.train.Saver()
        return Model(graph, batch_size, eval_batch_size, tf_train_dataset, tf_train_labels, eval_dataset, dropout_keep_probability, logits, loss, 
                     optimizer, train_prediction, eval_prediction, saver, global_step)
