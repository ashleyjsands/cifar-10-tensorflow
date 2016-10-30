"""
This file contains previous models.
"""
#TODO: add proper import statements.

class Model():
    
    def __init__(self, graph, batch_size, eval_batch_size, tf_train_dataset, tf_train_labels, eval_dataset, dropout_keep_probability, logits,
                 loss, optimizer, train_prediction, eval_prediction, saver, global_step, layer_weights):
        self.graph = graph
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.tf_train_dataset = tf_train_dataset
        self.tf_train_labels = tf_train_labels
        self.eval_dataset = eval_dataset
        self.dropout_keep_probability = dropout_keep_probability
        self.logits = logits
        self.loss = loss
        self.optimizer = optimizer
        self.train_prediction = train_prediction
        self.eval_prediction = eval_prediction
        self.saver = saver
        self.global_step = global_step
        self.layer_weights = layer_weights
        self.session = None

def create_same_padding_3_conv_one_hidden_model(learning_rate = 0.05, initialised_weights_stddev = 0.1, feature_maps = 16, number_of_hidden_neurons = 64, batch_size = 32, l2_lambda = 0.1, decay_steps = 10000, decay_rate = 0.96):
    patch_size = 5
    graph = tf.Graph()
    with graph.as_default():

        # Input data.
        tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))
        tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
        tf_valid_dataset = tf.constant(valid_dataset)
        tf_test_dataset = tf.constant(test_dataset)
        dropout_keep_probability = tf.placeholder(tf.float32)
        
        # Variables
        layer1_weights = tf.Variable(tf.truncated_normal(
            [patch_size, patch_size, num_channels, feature_maps], stddev=initialised_weights_stddev))
        layer1_biases = tf.Variable(tf.zeros([feature_maps]))

        layer2_weights = tf.Variable(tf.truncated_normal(
            [patch_size, patch_size, feature_maps, feature_maps], stddev=initialised_weights_stddev))
        layer2_biases = tf.Variable(tf.constant(initialised_weights_stddev * 10, shape=[feature_maps]))


        conv_layer3_weights = tf.Variable(tf.truncated_normal(
            [patch_size, patch_size, feature_maps, feature_maps], stddev=initialised_weights_stddev))
        conv_layer3_biases = tf.Variable(tf.constant(initialised_weights_stddev * 10, shape=[feature_maps]))

        #layel3_weights = tf.Variable(tf.truncated_normal(
        #    [image_size / 4 * image_size / 4 * feature_maps, number_of_hidden_neurons], stddev=initialised_weights_stddev))
        number_of_conv_layers = 3
        layer3_weights = tf.Variable(tf.truncated_normal(
            [int(math.ceil(image_size / (2.0 ** number_of_conv_layers)) * math.ceil(image_size / (2.0 ** number_of_conv_layers)) * feature_maps), number_of_hidden_neurons], stddev=initialised_weights_stddev))
        layer3_biases = tf.Variable(tf.constant(initialised_weights_stddev * 10, shape=[number_of_hidden_neurons]))


        layer4_weights = tf.Variable(tf.truncated_normal(
            [number_of_hidden_neurons, num_labels], stddev=initialised_weights_stddev))
        layer4_biases = tf.Variable(tf.constant(initialised_weights_stddev * 10, shape=[num_labels]))
        
        

        # Model.
        def create_model_graph(data, add_dropout = False):
            conv = tf.nn.conv2d(data, layer1_weights, [1, 1, 1, 1], padding='SAME')
            relu = tf.nn.relu(conv + layer1_biases)
            hidden = tf.nn.max_pool(relu, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

            conv = tf.nn.conv2d(hidden, layer2_weights, [1, 1, 1, 1], padding='SAME')
            relu = tf.nn.relu(conv + layer2_biases)
            hidden = tf.nn.max_pool(relu, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

            conv = tf.nn.conv2d(hidden, conv_layer3_weights, [1, 1, 1, 1], padding='SAME')
            relu = tf.nn.relu(conv + conv_layer3_biases)
            hidden = tf.nn.max_pool(relu, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

            shape = hidden.get_shape().as_list()
            reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
            hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
            if add_dropout:
                hidden = tf.nn.dropout(hidden, dropout_keep_probability)
            return tf.matmul(hidden, layer4_weights) + layer4_biases

        # Training computation.
        logits = create_model_graph(tf_train_dataset, add_dropout = True)
        layer_weights = [layer1_weights, layer2_weights, conv_layer3_weights, layer3_weights, layer4_weights]
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels) + get_l2_loss(l2_lambda, layer_weights))

        # Optimizer.
        global_step = tf.Variable(0, name="global_step", trainable=False)
        decayed_learning_rate = tf.train.exponential_decay(learning_rate, global_step, decay_steps, decay_rate)
        optimizer = tf.train.GradientDescentOptimizer(decayed_learning_rate).minimize(loss, global_step=global_step)

        # Predictions for the training, validation, and test data.
        train_prediction = tf.nn.softmax(logits)
        valid_prediction = tf.nn.softmax(create_model_graph(tf_valid_dataset))
        test_prediction = tf.nn.softmax(create_model_graph(tf_test_dataset))
        
        return Model(graph, batch_size, tf_train_dataset, tf_train_labels, tf_valid_dataset, tf_test_dataset, dropout_keep_probability, logits, loss, optimizer, train_prediction, valid_prediction, test_prediction)

def create_cv_cv_mp_cv_cv_mp_one_hidden_model(learning_rate = 0.05, initialised_weights_stddev = 0.1, feature_maps = 16, number_of_hidden_neurons = 64, batch_size = 32, l2_lambda = 0.1, decay_steps = 10000, decay_rate = 0.96):
    patch_size = 5
    graph = tf.Graph()
    with graph.as_default():

        # Input data.
        tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))
        tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
        tf_valid_dataset = tf.constant(valid_dataset)
        tf_test_dataset = tf.constant(test_dataset)
        dropout_keep_probability = tf.placeholder(tf.float32)
        
        # Variables
        layer1_weights = tf.Variable(tf.truncated_normal(
            [patch_size, patch_size, num_channels, feature_maps], stddev=initialised_weights_stddev))
        layer1_biases = tf.Variable(tf.zeros([feature_maps]))

        layer2_weights = tf.Variable(tf.truncated_normal(
            [patch_size, patch_size, feature_maps, feature_maps], stddev=initialised_weights_stddev))
        layer2_biases = tf.Variable(tf.constant(initialised_weights_stddev * 10, shape=[feature_maps]))


        conv_layer3_weights = tf.Variable(tf.truncated_normal(
            [patch_size, patch_size, feature_maps, feature_maps], stddev=initialised_weights_stddev))
        conv_layer3_biases = tf.Variable(tf.constant(initialised_weights_stddev * 10, shape=[feature_maps]))
        
        conv_layer4_weights = tf.Variable(tf.truncated_normal(
            [patch_size, patch_size, feature_maps, feature_maps], stddev=initialised_weights_stddev))
        conv_layer4_biases = tf.Variable(tf.constant(initialised_weights_stddev * 10, shape=[feature_maps]))

        number_of_max_pool_layers = 2
        conv_output_size = int(math.ceil(image_size / (2.0 ** number_of_max_pool_layers)) * math.ceil(image_size / (2.0 ** number_of_max_pool_layers)) * feature_maps)
        #print "conv_output_size %s" % conv_output_size
        layer3_weights = tf.Variable(tf.truncated_normal(
            [conv_output_size, number_of_hidden_neurons], stddev=initialised_weights_stddev))
        layer3_biases = tf.Variable(tf.constant(initialised_weights_stddev * 10, shape=[number_of_hidden_neurons]))


        layer4_weights = tf.Variable(tf.truncated_normal(
            [number_of_hidden_neurons, num_labels], stddev=initialised_weights_stddev))
        layer4_biases = tf.Variable(tf.constant(initialised_weights_stddev * 10, shape=[num_labels]))

        # Model.
        def create_model_graph(data, add_dropout = False):
            conv = tf.nn.conv2d(data, layer1_weights, [1, 1, 1, 1], padding='SAME')
            hidden = tf.nn.relu(conv + layer1_biases)
            shape = hidden.get_shape().as_list()
            #print "hidden shape: %s" % shape

            conv = tf.nn.conv2d(hidden, layer2_weights, [1, 1, 1, 1], padding='SAME')
            relu = tf.nn.relu(conv + layer2_biases)
            hidden = tf.nn.max_pool(relu, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
            shape = hidden.get_shape().as_list()
            #print "hidden shape: %s" % shape

            conv = tf.nn.conv2d(hidden, conv_layer3_weights, [1, 1, 1, 1], padding='SAME')
            hidden = tf.nn.relu(conv + conv_layer3_biases)
            
            conv = tf.nn.conv2d(hidden, conv_layer4_weights, [1, 1, 1, 1], padding='SAME')
            relu = tf.nn.relu(conv + conv_layer4_biases)
            hidden = tf.nn.max_pool(relu, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
            shape = hidden.get_shape().as_list()
            #print "hidden shape: %s" % shape

            shape = hidden.get_shape().as_list()
            #print "hidden shape: %s" % shape
            reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
            hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
            if add_dropout:
                hidden = tf.nn.dropout(hidden, dropout_keep_probability)
            return tf.matmul(hidden, layer4_weights) + layer4_biases

        # Training computation.
        logits = create_model_graph(tf_train_dataset, add_dropout = True)
        layer_weights = [layer1_weights, layer2_weights, conv_layer3_weights, conv_layer4_weights, layer3_weights, layer4_weights]
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels) + get_l2_loss(l2_lambda, layer_weights))

        # Optimizer.
        global_step = tf.Variable(0, name="global_step", trainable=False)
        decayed_learning_rate = tf.train.exponential_decay(learning_rate, global_step, decay_steps, decay_rate)
        optimizer = tf.train.GradientDescentOptimizer(decayed_learning_rate).minimize(loss, global_step=global_step)

        # Predictions for the training, validation, and test data.
        train_prediction = tf.nn.softmax(logits)
        valid_prediction = tf.nn.softmax(create_model_graph(tf_valid_dataset))
        test_prediction = tf.nn.softmax(create_model_graph(tf_test_dataset))
        
        return Model(graph, batch_size, tf_train_dataset, tf_train_labels, tf_valid_dataset, tf_test_dataset, dropout_keep_probability, logits, loss, optimizer, train_prediction, valid_prediction, test_prediction)
    

def create_three_double_conv_layers_one_hidden_model(learning_rate = 0.05, initialised_weights_stddev = 0.1, feature_maps = 16, number_of_hidden_neurons = 64, batch_size = 32, l2_lambda = 0.1, decay_steps = 10000, decay_rate = 0.96):
    patch_size = 5
    graph = tf.Graph()
    with graph.as_default():

        # Input data.
        tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))
        tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
        tf_valid_dataset = tf.constant(valid_dataset)
        tf_test_dataset = tf.constant(test_dataset)
        dropout_keep_probability = tf.placeholder(tf.float32)
        
        # Variables
        layer1_weights = tf.Variable(tf.truncated_normal(
            [patch_size, patch_size, num_channels, feature_maps], stddev=initialised_weights_stddev))
        layer1_biases = tf.Variable(tf.zeros([feature_maps]))

        layer2_weights = tf.Variable(tf.truncated_normal(
            [patch_size, patch_size, feature_maps, feature_maps], stddev=initialised_weights_stddev))
        layer2_biases = tf.Variable(tf.constant(initialised_weights_stddev * 10, shape=[feature_maps]))

        conv_layer3_weights = tf.Variable(tf.truncated_normal(
            [patch_size, patch_size, feature_maps, feature_maps], stddev=initialised_weights_stddev))
        conv_layer3_biases = tf.Variable(tf.constant(initialised_weights_stddev * 10, shape=[feature_maps]))
        
        conv_layer4_weights = tf.Variable(tf.truncated_normal(
            [patch_size, patch_size, feature_maps, feature_maps], stddev=initialised_weights_stddev))
        conv_layer4_biases = tf.Variable(tf.constant(initialised_weights_stddev * 10, shape=[feature_maps]))
        
        conv_layer5_weights = tf.Variable(tf.truncated_normal(
            [patch_size, patch_size, feature_maps, feature_maps], stddev=initialised_weights_stddev))
        conv_layer5_biases = tf.Variable(tf.constant(initialised_weights_stddev * 10, shape=[feature_maps]))
        
        conv_layer6_weights = tf.Variable(tf.truncated_normal(
            [patch_size, patch_size, feature_maps, feature_maps], stddev=initialised_weights_stddev))
        conv_layer6_biases = tf.Variable(tf.constant(initialised_weights_stddev * 10, shape=[feature_maps]))

        number_of_max_pool_layers = 3
        conv_output_size = int(math.ceil(image_size / (2.0 ** number_of_max_pool_layers)) * math.ceil(image_size / (2.0 ** number_of_max_pool_layers)) * feature_maps)
        #print "conv_output_size %s" % conv_output_size
        layer3_weights = tf.Variable(tf.truncated_normal(
            [conv_output_size, number_of_hidden_neurons], stddev=initialised_weights_stddev))
        layer3_biases = tf.Variable(tf.constant(initialised_weights_stddev * 10, shape=[number_of_hidden_neurons]))


        layer4_weights = tf.Variable(tf.truncated_normal(
            [number_of_hidden_neurons, num_labels], stddev=initialised_weights_stddev))
        layer4_biases = tf.Variable(tf.constant(initialised_weights_stddev * 10, shape=[num_labels]))

        # Model.
        def create_model_graph(data, add_dropout = False):
            conv = tf.nn.conv2d(data, layer1_weights, [1, 1, 1, 1], padding='SAME')
            hidden = tf.nn.relu(conv + layer1_biases)
            #shape = hidden.get_shape().as_list()
            #print "hidden shape: %s" % shape

            conv = tf.nn.conv2d(hidden, layer2_weights, [1, 1, 1, 1], padding='SAME')
            relu = tf.nn.relu(conv + layer2_biases)
            hidden = tf.nn.max_pool(relu, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
            #shape = hidden.get_shape().as_list()
            #print "hidden shape: %s" % shape

            conv = tf.nn.conv2d(hidden, conv_layer3_weights, [1, 1, 1, 1], padding='SAME')
            hidden = tf.nn.relu(conv + conv_layer3_biases)
            
            conv = tf.nn.conv2d(hidden, conv_layer4_weights, [1, 1, 1, 1], padding='SAME')
            relu = tf.nn.relu(conv + conv_layer4_biases)
            hidden = tf.nn.max_pool(relu, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
            #shape = hidden.get_shape().as_list()
            #print "hidden shape: %s" % shape
            
            conv = tf.nn.conv2d(hidden, conv_layer5_weights, [1, 1, 1, 1], padding='SAME')
            hidden = tf.nn.relu(conv + conv_layer5_biases)
            
            conv = tf.nn.conv2d(hidden, conv_layer6_weights, [1, 1, 1, 1], padding='SAME')
            relu = tf.nn.relu(conv + conv_layer6_biases)
            hidden = tf.nn.max_pool(relu, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
            #shape = hidden.get_shape().as_list()
            #print "hidden shape: %s" % shape

            shape = hidden.get_shape().as_list()
            #print "hidden shape: %s" % shape
            reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
            hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
            if add_dropout:
                hidden = tf.nn.dropout(hidden, dropout_keep_probability)
            return tf.matmul(hidden, layer4_weights) + layer4_biases

        # Training computation.
        logits = create_model_graph(tf_train_dataset, add_dropout = True)
        layer_weights = [layer1_weights, layer2_weights, conv_layer3_weights, conv_layer4_weights, conv_layer5_weights, conv_layer6_weights, layer3_weights, layer4_weights]
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels) + get_l2_loss(l2_lambda, layer_weights))

        # Optimizer.
        global_step = tf.Variable(0, name="global_step", trainable=False)
        decayed_learning_rate = tf.train.exponential_decay(learning_rate, global_step, decay_steps, decay_rate)
        optimizer = tf.train.GradientDescentOptimizer(decayed_learning_rate).minimize(loss, global_step=global_step)

        # Predictions for the training, validation, and test data.
        train_prediction = tf.nn.softmax(logits)
        valid_prediction = tf.nn.softmax(create_model_graph(tf_valid_dataset))
        test_prediction = tf.nn.softmax(create_model_graph(tf_test_dataset))
        
        return Model(graph, batch_size, tf_train_dataset, tf_train_labels, tf_valid_dataset, tf_test_dataset, dropout_keep_probability, logits, loss, optimizer, train_prediction, valid_prediction, test_prediction)    

from neural_network import depth_concat

def create_naive_inception_module_model(learning_rate = 0.05, initialised_weights_stddev = 0.1, feature_maps = 16, batch_size = 32, eval_batch_size = 100, l2_lambda = 0.1, decay_steps = 10000, decay_rate = 0.96):
    graph = tf.Graph()
    with graph.as_default():

        # Input data.
        tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))
        tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
        eval_dataset = tf.placeholder(tf.float32, shape=(eval_batch_size, image_size, image_size, num_channels))
        dropout_keep_probability = tf.placeholder(tf.float32)
        
        # In the naive inception module, we have 6 layers: the input layer, followed by the 1x1 conv, 3x3 conv, 5x5 conv
        # and 3x3 maxpooling layer and lastly the DepthConcat layer.
        
        patch_size = 1
        one_by_one_conv_weights = tf.Variable(tf.truncated_normal(
            [patch_size, patch_size, num_channels, feature_maps], stddev=initialised_weights_stddev))
        one_by_one_conv_biases = tf.Variable(tf.constant(initialised_weights_stddev * 10, shape=[feature_maps]))
        
        patch_size = 3
        three_by_three_conv_weights = tf.Variable(tf.truncated_normal(
            [patch_size, patch_size, num_channels, feature_maps], stddev=initialised_weights_stddev))
        three_by_three_conv_biases = tf.Variable(tf.constant(initialised_weights_stddev * 10, shape=[feature_maps]))
        
        patch_size = 5
        five_by_five_conv_weights = tf.Variable(tf.truncated_normal(
            [patch_size, patch_size, num_channels, feature_maps], stddev=initialised_weights_stddev))
        five_by_five_conv_biases = tf.Variable(tf.constant(initialised_weights_stddev * 10, shape=[feature_maps]))
    
        # The 3x3 maxpooling layer, DepthConcat layer, and averagepooling layer don't need any variables.
        
        # Now a fully connected layer
        number_of_adjacent_layers = 4
        num_conv_layers = 3
        depth_concat_depth = feature_maps * num_conv_layers + num_channels # num_channels is the 3x2 maxpooling depth.
        # I expect avg_pool_ouput to have a shape of (batch_size, 1, 1, depth_concat_depth)
        # WARNING: I may have gotten the fc_weights tensor size wrong.
        fc_weights = tf.Variable(tf.truncated_normal(
            [depth_concat_depth, num_labels], stddev=initialised_weights_stddev))
        #fc_biases = tf.Variable(tf.constant(initialised_weights_stddev * 10, shape=[num_labels]))
        fc_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))
        
        # Model.
        def create_model_graph(data, add_dropout = False):
            shape = data.get_shape().as_list()
            #print "data shape: %s" % shape
            
            conv = tf.nn.conv2d(data, one_by_one_conv_weights, [1, 1, 1, 1], padding='SAME')
            one_by_one_output = tf.nn.relu(conv + one_by_one_conv_biases)
            shape = one_by_one_output.get_shape().as_list()
            #print "one_by_one_output shape: %s" % shape

            conv = tf.nn.conv2d(data, three_by_three_conv_weights, [1, 1, 1, 1], padding='SAME')
            three_by_three_output = tf.nn.relu(conv + three_by_three_conv_biases)
            shape = three_by_three_output.get_shape().as_list()
            #print "three_by_three_output shape: %s" % shape

            conv = tf.nn.conv2d(data, five_by_five_conv_weights, [1, 1, 1, 1], padding='SAME')
            five_by_five_output = tf.nn.relu(conv + five_by_five_conv_biases)
            shape = five_by_five_output.get_shape().as_list()
            #print "five_by_five_output shape: %s" % shape
            
            max_pool_output = tf.nn.max_pool(data, [1, 3, 3, 1], [1, 1, 1, 1], padding='SAME')
            shape = max_pool_output.get_shape().as_list()
            #print "max_pool_output shape: %s" % shape
            
            #print([one_by_one_output, three_by_three_output, five_by_five_output, max_pool_output])
            depth_concat_output = depth_concat([one_by_one_output, three_by_three_output, five_by_five_output, max_pool_output])
            shape = depth_concat_output.get_shape().as_list()
            #print "depth_concat_output shape: %s" % shape
            
            # The patch size of the avg_pool must match the patch_size of the depth_concat_output
            # I assume that the padding must be VALID based on Google's white paper: http://arxiv.org/pdf/1409.4842v1.pdf
            largest_patch_size = 32 # This is the height/width of depth_concat_output
            avg_pool_output = tf.nn.avg_pool(depth_concat_output, [1, largest_patch_size, largest_patch_size, 1], [1, 1, 1, 1], padding='VALID', name=None)
            shape = avg_pool_output.get_shape().as_list()
            #print "avg_pool_output shape: %s" % shape

            # Flatten the average_pool_output from 4 dimensions down to 2.
            batch_index = 0
            reshape_tensor = tf.reshape(avg_pool_output, (data.get_shape().as_list()[batch_index], 1 * 1 * depth_concat_depth))
            #print "reshape_tensor shape: %s" % reshape_tensor.get_shape().as_list()
            
            # TODO: add dropout.
            #if add_dropout:
            #    hidden = tf.nn.dropout(hidden, dropout_keep_probability)
            return tf.matmul(reshape_tensor, fc_weights) + fc_biases

        # Training computation.
        logits = create_model_graph(tf_train_dataset, add_dropout = True)
        layer_weights = [one_by_one_conv_weights, three_by_three_conv_weights, five_by_five_conv_weights, fc_weights]
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
