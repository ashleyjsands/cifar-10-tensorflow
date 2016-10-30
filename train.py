import time
import numpy as np
import tensorflow as tf

from data import num_labels


def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

def train_model_in_batches(model, datasets, steps, dropout_keep_prob, load_model = False):
    batch_size = model.batch_size
    start_time = time.time()
    steps_to_training_accuracies = {}
    steps_to_validation_predictions = {}
    with tf.Session(graph=model.graph) as session:
        model.session = session # Save the session for future visualisation use.
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
        learning_check_step = 500
        minimum_validation_step_size = 1000
        validation_step_size = int(max(steps / 100, minimum_validation_step_size))
        save_step_size = 50000
        untrained_validation_accuracy = (100 / num_labels) * 1.2
        premature_stop_steps_minimum = 3000
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
            
            if (step % validation_step_size == 0) or step == learning_check_step:
                training_accuracy = accuracy(predictions, batch_labels)
                steps_to_training_accuracies[step] = training_accuracy
                validation_predictions = eval_predictions(session, model, datasets.valid_dataset, datasets.valid_labels)
                #print "validation_predictions shape: ", validation_predictions.shape, " valid_labels shape: ", datasets.valid_labels.shape
                steps_to_validation_predictions[step] = validation_predictions
                valdiation_accuracy = accuracy(validation_predictions, datasets.valid_labels)
                print("step:", step, "minibatch loss:", l, "minibatch accuracy: %.1f%%" % training_accuracy, "validation accuracy: %.1f%%" % valdiation_accuracy)
                if valdiation_accuracy < untrained_validation_accuracy and step >= premature_stop_steps_minimum:
                    print("Premature stop due to low validation accuracy.")
                    return steps_to_training_accuracies, steps_to_validation_predictions
            if step % save_step_size == 0:
                save_path = model.saver.save(session, "./%s.ckpt" % model_name, global_step=model.global_step)
        save_path = model.saver.save(session, "./%s.ckpt" % model_name, global_step=model.global_step)
        test_predictions = eval_predictions(session, model, datasets.test_dataset, datasets.test_labels)
        print("Test accuracy at step %s: %.1f%%\n" % (step, accuracy(test_predictions, datasets.test_labels)))
        seconds_in_an_hour = 60 * 60
        print("Elapsed time: %s hours" % ((time.time() - start_time) / seconds_in_an_hour))
        return steps_to_training_accuracies, steps_to_validation_predictions

def eval_predictions(session, model, dataset, labels):
    dataset_size = dataset.shape[0]
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
    return predictions
