"""Routine for decoding the ocr tfrecords file format."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import glob

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

# Process images of this size. Note that this differs from the original CIFAR
# image size of 32 x 32. If one alters this number, then the entire model
# architecture will change and any model would need to be retrained.
IMAGE_HEIGHT = 48
IMAGE_WIDTH = 152
IMAGE_DEPTH = 3

# Global constants describing the ocr data set.
NUM_CLASSES = 0
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 320000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 6271

MIN_FRACTION_OF_EXAMPLES_IN_QUEUE = 0.2


def get_num_classes():
    with open("./data/codec.txt") as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    return len(content) + 1 + 1  # first 1 for tf's sequence zero-padding and second is for blank label


NUM_CLASSES = get_num_classes()

print("NUM_CLASSES " + str(NUM_CLASSES))


def read_serialized(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    return serialized_example


def batch_serialized(serialized_example, batch_size, min_queue_examples, shuffle=True):
    num_preprocess_threads = 8
    if shuffle:
        serialized_examples_batch = tf.train.shuffle_batch(
            [serialized_example],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples)
    else:
        serialized_examples_batch = tf.train.batch(
            [serialized_example],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size)

    return serialized_examples_batch


def parse_serialized_examples_batch(serialized_examples_batch, batch_size):
    feature_to_tensor = {
        'image': tf.FixedLenFeature([], tf.string),
        'height': tf.FixedLenFeature([1], tf.int64),
        'width': tf.FixedLenFeature([1], tf.int64),
        'label': tf.VarLenFeature(tf.int64),
        'label_length': tf.FixedLenFeature([1], tf.int64)
    }
    features = tf.parse_example(serialized_examples_batch, feature_to_tensor)

    class ocrRecord(object):
        pass

    result = ocrRecord()

    result.heights = tf.cast(features['height'], tf.int32)
    result.widths = tf.cast(features['width'], tf.int32)
    result.depth = 1

    # shape_1d = result.height * result.width * result.depth
    shape_1d = IMAGE_HEIGHT * IMAGE_WIDTH * IMAGE_DEPTH

    def decode_image_string(string):
        decoded_image = tf.decode_raw(string, tf.uint8)
        return tf.cast(decoded_image, tf.uint8)

    imgs_1d = tf.map_fn(decode_image_string, features['image'], dtype=tf.uint8,
                        back_prop=False, parallel_iterations=15)

    imgs_1d = tf.reshape(imgs_1d, [batch_size, shape_1d])
    imgs_1d.set_shape([batch_size, shape_1d])

    result.uint8images = tf.reshape(imgs_1d, [batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH])
    result.uint8images.set_shape([batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH])

    result.label_lengths = tf.cast(features['label_length'], tf.int32)
    result.label_lengths = tf.reshape(result.label_lengths, [batch_size])
    result.label_lengths.set_shape([batch_size])

    result.labels = tf.cast(features['label'], tf.int32)

    # Convert for timestep input
    result.uint8image = tf.transpose(result.uint8images, [0, 2, 1, 3])
    return result


def _generate_image_and_label_batch(image, label, seq_len, min_queue_examples,
                                    batch_size, shuffle):
    """Construct a queued batch of images and labels.
  
    Args:
      image: 3-D Tensor of [height, width, 3] of type.float32.
      label: 1-D Tensor of type.int32
      min_queue_examples: int32, minimum number of samples to retain
        in the queue that provides of batches of examples.
      batch_size: Number of images per batch.
      shuffle: boolean indicating whether to use a shuffling queue.
  
    Returns:
      images: Images. 4D tensor of [batch_size, height, width, 3] size.
      labels: Labels. 1D tensor of [batch_size] size.
    """
    # Create a queue that shuffles the examples, and then
    # read 'batch_size' images + labels from the example queue.
    num_preprocess_threads = 8
    if shuffle:
        images, label_batch, seq_len_batch = tf.train.shuffle_batch(
            [image, label, seq_len],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples)
    else:
        images, label_batch, seq_len_batch = tf.train.batch(
            [image, label, seq_len],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size)
    return images, label_batch, seq_len_batch


def distorted_inputs(data_dir, batch_size):
    filenames = tf.train.match_filenames_once(os.path.abspath(os.path.join(data_dir, "*.tfrecords")))

    print("TFRecords filenames:")
    print(filenames)
    # Create a queue that produces the filenames to read.
    filename_queue = tf.train.string_input_producer(filenames)

    # Read examples from files in the filename queue.
    serialized_example = read_serialized(filename_queue)

    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN * min_fraction_of_examples_in_queue)

    serialized_examples_batch = batch_serialized(serialized_example, batch_size, min_queue_examples, shuffle=True)

    batch_result = parse_serialized_examples_batch(serialized_examples_batch, batch_size)

    float_images = tf.cast(batch_result.uint8image, tf.float32)

    float_images = tf.cast(float_images, tf.float32) * (1. / 255) - 0.5
    float_images.set_shape([batch_size, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_DEPTH])

    # Because these operations are not commutative, consider randomizing
    # the order their operation.
    def augment(input_image):
        #   input_image = tf.image.per_image_standardization(input_image)
        # input_image = tf.image.random_brightness(input_image, max_delta=0.3)
        # input_image = tf.image.random_contrast(input_image, lower=0.3, upper=1.7)
        input_image = tf.image.random_hue(input_image, max_delta=0.5)
        # input_image = tf.image.random_saturation(input_image, lower=0.0, upper=0.9)
        input_image = tf.minimum(input_image, 1.0)
        input_image = tf.maximum(input_image, -1.0)
        return input_image

    float_images = tf.map_fn(augment, float_images, dtype=tf.float32,
                             back_prop=False, parallel_iterations=64)

    # Ensure that the random shuffling has good mixing properties.

    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                             MIN_FRACTION_OF_EXAMPLES_IN_QUEUE)
    print('Filling queue with %d line images before starting to train. '
          'This will take a few minutes.' % min_queue_examples)

    # Adding images to summary
    tf.summary.image('images_train', float_images, max_outputs=8)

    # Generate a batch of images and labels by building up a queue of examples.
    return float_images, batch_result.labels, batch_result.label_lengths


def inputs(data_dir, batch_size):
    """Construct input for evaluation data using the Reader ops.
  
    Args:
      eval_data: bool, indicating if one should use the train or eval data set.
      data_dir: Path to the data directory.
      batch_size: Number of images per batch.
  
    Returns:
      images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
      labels: Labels. 1D tensor of [batch_size] size.
    """
    filenames = glob.glob(os.path.abspath(os.path.join(data_dir, "*.tfrecords")))

    print("TFRecords filenames:")
    print(filenames)
    # Create a queue that produces the filenames to read.
    filename_queue = tf.train.string_input_producer(filenames)

    # Read examples from files in the filename queue.
    serialized_example = read_serialized(filename_queue)

    min_fraction_of_examples_in_queue = 0.2
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN * min_fraction_of_examples_in_queue)

    serialized_examples_batch = batch_serialized(serialized_example, batch_size, min_queue_examples, shuffle=False)

    batch_result = parse_serialized_examples_batch(serialized_examples_batch, batch_size)

    reshaped_images = tf.cast(batch_result.uint8image, tf.float32)

    # Image processing for training the network. Note the many random
    # distortions applied to the image.

    float_images = tf.cast(reshaped_images, tf.float32) * (1. / 255)  # - 0.5
    float_images.set_shape([batch_size, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_DEPTH])

    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                             MIN_FRACTION_OF_EXAMPLES_IN_QUEUE)
    print('Filling queue with %d CIFAR images before starting to train. '
          'This will take a few minutes.' % min_queue_examples)

    # Adding images to summary
    tf.summary.image('images_val', float_images, max_outputs=4)

    # Generate a batch of images and labels by building up a queue of examples.
    return float_images, batch_result.labels, batch_result.label_lengths
