from data import get_datasets_and_labels, reformat, Datasets, image_size, num_labels, num_channels
from visual import visualise_accuracies
from inception_module_model import create_inception_module_model
from train import accuracy, train_model_in_batches, eval_predictions

train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = get_datasets_and_labels()

print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

print('\nReformatting datasets')
train_dataset, train_labels = reformat(train_dataset, train_labels, image_size, num_channels, num_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels, image_size, num_channels, num_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels, image_size, num_channels, num_labels)

datasets = Datasets(train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels)

print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

if __name__ == "__main__":
    #run_on_cpu= True
    #if run_on_cpu:
        #CUDA_VISIBLE_DEVICES = ''
    # 5K steps for GPU test
    model = create_inception_module_model(learning_rate = 0.00025, eval_batch_size=100, l2_lambda = 0.025, pre_layer_feature_maps = 128, feature_maps = 64, initialised_weights_stddev = 0.25, decay_steps = 5000, decay_rate = 0.96)
    steps_to_validation_predictions = train_model_in_batches(model, datasets, 5001, dropout_keep_prob = 0.9, load_model = False)
    correct_prediction_indexes, incorrect_prediction_indexes = visualise_accuracies(steps_to_validation_predictions, valid_labels)
