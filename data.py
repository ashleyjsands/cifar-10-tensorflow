import os, pickle, tarfile, ntpath
import numpy as np

data_dir = "."
image_size = 32
num_labels = 10
num_channels = 3 # RGB

import functools

def maybe_download():
    dest_directory = data_dir
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
                float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, reporthook=_progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')   
    
def load_cifar_10_dataset():
    print("Opening CIFAR 10 dataset")
    dataset = {}
    with tarfile.open(data_dir + "/cifar-10-python.tar.gz", "r:gz") as tar:
        for member in tar.getmembers():
            if member.isfile():
                if "_batch" in member.name:
                    file_name = ntpath.basename(member.name)
                    f = tar.extractfile(member)
                    batch_dataset = pickle.load(f, encoding='latin1') 
                    dataset[file_name] = batch_dataset
                elif member.name.endswith("batches.meta"):
                    f = tar.extractfile(member)
                    label_names = pickle.load(f, encoding='latin1') 
                    dataset["meta"] = label_names
    print("Finished opening CIFAR 10 dataset")
    return dataset
     
def merge_datasets(dataset_one, dataset_two):
    return {
        "data": np.concatenate((dataset_one["data"], dataset_two["data"])),
        "labels": dataset_one["labels"] + dataset_two["labels"], 
    }

def get_merged_training_datasets(dataset_batches_dict):
    training_dataset_names = [ "data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4" ]
    training_datasets = map(lambda name: dataset_batches_dict[name], training_dataset_names)
    training_dataset_and_labels = functools.reduce(merge_datasets, training_datasets)
    validation_dataset_and_labels = dataset_batches_dict["data_batch_5"]
    test_dataset_and_labels = dataset_batches_dict["test_batch"]
    return (
        np.asarray(training_dataset_and_labels["data"]), np.asarray(training_dataset_and_labels["labels"]),
        np.asarray(validation_dataset_and_labels["data"]), np.asarray(validation_dataset_and_labels["labels"]),
        np.asarray(test_dataset_and_labels["data"]), np.asarray(test_dataset_and_labels["labels"])
    )

def get_datasets_and_labels():
    maybe_download()
    dataset_batches_dict = load_cifar_10_dataset()
    label_names = dataset_batches_dict["meta"]["label_names"]
    return get_merged_training_datasets(dataset_batches_dict)

def reformat(dataset, labels, image_size, num_channels, num_labels):
    """
    Reformat into a TensorFlow-friendly shape:
    - convolutions need the image data formatted as a cube (width by height by #channels)
    - labels as float 1-hot encodings.
    """
    #dataset = dataset.reshape(
    #  (-1, image_size, image_size, num_channels)).astype(np.float32)
    
    # the dataset is of a shape (*, num_channels * image_size * image_size) 
    # with the red values first, followed by the green, then blue.
    dataset = dataset
    x = dataset.reshape((-1, num_channels, image_size * image_size)) # break the channels into their own axes.
    y = x.transpose([0, 2, 1]) # This transpose the matrix by swapping the second and third axes, but not the first. This puts matching RGB values together
    reformated_dataset = y.reshape((-1, image_size, image_size, num_channels)).astype(np.float32) # Turn the dataset into a 4D tensor of a collection of images, with axes of width, height and colour channels.
    labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
    return reformated_dataset, labels

class Datasets:

    def __init__(self, train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels):
        self.train_dataset = train_dataset 
        self.train_labels = train_labels 
        self.valid_dataset = valid_dataset
        self.valid_labels = valid_labels
        self.test_dataset = test_dataset
        self.test_labels = test_labels

def subtract_mean_from_images(datasets):
    """The purpose of subtracting the mean from the images is to "centre" the values of the dataset.
       Doing so allows the weights to not grow out of control and thus allow the model to train quicker and produce superior performance."""
    # I assume that its a case of data leakage to calculate the mean using the test dataset in addition to the rest.
    mean = get_mean_of_training_and_validation_images(datasets)
    print("Subtracting mean of", mean, "from all images.")
    datasets.train_dataset = subtract_value_from_dataset(mean, datasets.train_dataset)
    datasets.valid_dataset = subtract_value_from_dataset(mean, datasets.valid_dataset)
    datasets.test_dataset = subtract_value_from_dataset(mean, datasets.test_dataset)

def get_mean_of_training_and_validation_images(datasets):
    total = sum_dataset(datasets.train_dataset) + sum_dataset(datasets.valid_dataset)
    number_of_images = (len(datasets.train_dataset) + len(datasets.valid_dataset))

    from data import image_size, num_channels

    return int(round(total / (number_of_images * image_size * image_size * num_channels)))

def sum_dataset(dataset):
    total = 0
    for image in dataset:
        total += np.sum(image)
    return total

def subtract_value_from_dataset(value, dataset):
    return [image - value for image in dataset]
