"""
This file contains previous models.
"""
#TODO: add proper import statements.

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

