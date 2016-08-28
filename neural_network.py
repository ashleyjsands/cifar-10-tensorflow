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

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

def train_model_in_batches(model, datasets, steps, dropout_keep_prob, load_model = False):
    batch_size = model.batch_size
    start_time = time.time()
    steps_to_validation_predictions = {}
    with tf.Session(graph=model.graph) as session:
        init_op = tf.initialize_all_variables()
        #saver = tf.train.Saver()
        session.run(init_op) # All variables must be initialised before the saver potentionally restores the checkpoint below.
        print("Initialized")
        model_name = 'model'
        if load_model:
            ckpt = tf.train.get_checkpoint_state("./")
            if ckpt and ckpt.model_checkpoint_path:
                model.saver.restore(session, ckpt.model_checkpoint_path)
                print("model loaded")
            else:
                raise Error("No checkpoint.")
        for step in range(steps):
            offset = (step * batch_size) % (datasets.train_labels.shape[0] - batch_size)
            batch_data = datasets.train_dataset[offset:(offset + batch_size), :, :, :]
            batch_labels = datasets.train_labels[offset:(offset + batch_size), :]
            feed_dict = {
                model.tf_train_dataset: batch_data,
                model.tf_train_labels : batch_labels,
                model.dropout_keep_probability: dropout_keep_prob
            }
            _, l, predictions = session.run(
                [model.optimizer, model.loss, model.train_prediction], feed_dict=feed_dict)
            if (step % 1000 == 0):
                training_accuracy = accuracy(predictions, batch_labels)
                validation_predictions = eval_predictions(session, model, datasets.valid_dataset, datasets.valid_labels)
                #print "validation_predictions shape: ", validation_predictions.shape, " valid_labels shape: ", datasets.valid_labels.shape
                steps_to_validation_predictions[step] = validation_predictions
                valdiation_accuracy = accuracy(validation_predictions, datasets.valid_labels)
                print("step:", step, "minibatch loss:", l, "minibatch accuracy: %.1f%%" % training_accuracy, "validation accuracy: %.1f%%" % valdiation_accuracy)
            if step % 50000 == 0:
                save_path = model.saver.save(session, "./%s.ckpt" % model_name, global_step=model.global_step)
        test_predictions = eval_predictions(session, model, datasets.test_dataset, datasets.test_labels)
        print("Test accuracy at step %s: %.1f%%\n" % (step, accuracy(test_predictions, datasets.test_labels)))
        seconds_in_an_hour = 60 * 60
        print("Elapsed time: %s hours" % ((time.time() - start_time) / seconds_in_an_hour))
        return steps_to_validation_predictions

def eval_predictions(session, model, dataset, labels):
    dataset_size = dataset.shape[0]
    num_labels_index = 1
    num_labels = labels.shape[num_labels_index]
    batch_size = model.eval_batch_size
    #print "dataset_size: ", dataset_size, " batch_size: ", batch_size
    if dataset_size % batch_size != 0:
        raise "batch_size must be a multiple of dataset_size."
    predictions = np.ndarray(shape=(dataset_size, num_labels), dtype=np.float32)
    steps = dataset_size // batch_size
    #print "steps: ", steps
    for step in range(steps):
        offset = (step * batch_size)
        #print "offset ", offset
        batch_data = dataset[offset:(offset + batch_size), :, :, :]
        feed_dict = {
            model.eval_dataset: batch_data,
        }
        #predictions[offset:offset+batch_size, :] = model.eval_prediction.eval(feed_dict)
        predictions[offset:offset+batch_size, :] = session.run(model.eval_prediction, feed_dict=feed_dict)
    #print predictions
    return predictions
