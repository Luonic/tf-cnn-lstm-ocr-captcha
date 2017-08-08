import tensorflow as tf
from ImageAugmenter import ImageAugmenter
import cv2
from tqdm import tqdm
import glob
import os
import numpy as np
from random import shuffle
from joblib import Parallel, delayed
import multiprocessing

num_cores = multiprocessing.cpu_count()
print("CPU count:" + str(num_cores))

NUM_EXAMPLES_PER_RECORD = 100000
MAX_LABEL_LENGTH = 4
MAX_IMAGE_WIDTH = 152
NUM_CLASSES = 0

codec = {}

img_size = (152, 48)  # WxH

# data_dir = "data_dbg"
data_dir = "data"


def load_codec():
    codec = {}
    with open("./data/codec.txt") as f:
        content = f.readlines()
    content = [x.strip("\n") for x in content]
    for idx, line in enumerate(content):
        for char in line:
            codec[str(char)] = idx + 1
    return codec


def get_num_classes():
    with open("./data/codec.txt") as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    return len(content) + 1 + 1  # first 1 for tf's sequence zero-padding and second is for blank label


def string_label_to_numbers(string_label):
    # TODO: WATCH THIS!
    string_label = string_label.lower()
    string_label = string_label.strip()
    num_label = []
    for idx in range(len(string_label)):
        if idx < len(string_label):
            char = string_label[idx]
            if char in codec:
                num_label.append(codec[char])
            else:
                num_label.append(0)
                print("Cannot find char " + str(char) + " in codec")
    return np.asarray(num_label, dtype=np.int64)


def numbers_to_string_label(numbers):
    decoded_label = ""
    for char_index in numbers:
        decoded_label += codec_index_to_char(char_index)
    return decoded_label


def codec_index_to_char(code):
    for dict_char, dict_index in codec.items():
        if code == dict_index:
            return dict_char


def is_image_valid(image):
    if type(image) is not np.ndarray:
        return False
    if image.shape[0] == 0:
        return False
    if image.shape[1] == 0:
        return False
    if cv2.mean(image) <= (1, 1, 1, 0):
        return False
    return True


# This function recives paths to images and lines from file with labels
# and returns only path to images that have corresponding label
def make_label(long_filename):
    try:
        base_filename = os.path.basename(long_filename)
        base_filename_no_ext = os.path.splitext(base_filename)[0]
        str_label = base_filename_no_ext[0] + base_filename_no_ext[1] + base_filename_no_ext[2] + \
                    base_filename_no_ext[3]
        num_label = string_label_to_numbers(str_label)
        return num_label
    except Exception as e:
        print("Bad label: " + str(long_filename))
        exit()


def get_distilled_labels(filenames):
    result_labels = []
    print("Creating labels")
    result_labels = Parallel(n_jobs=num_cores)(delayed(make_label)(long_filename) for long_filename in tqdm(filenames))
    return result_labels

# This function recives paths to images and lines from file with labels
# and returns only path to images that have corresponding label
def get_distilled_labels_slow(filenames):
    result_filenames = []
    result_labels = []
    print("Creating labels")
    for idx, long_filename in tqdm(enumerate(filenames)):
        base_filename = os.path.basename(long_filename)
        base_filename_no_ext = os.path.splitext(base_filename)[0]
        str_label = base_filename_no_ext[0] + base_filename_no_ext[1] + base_filename_no_ext[2] + base_filename_no_ext[3]
        num_label = string_label_to_numbers(str_label, 4)
        result_labels.append(num_label)
        result_filenames.append(long_filename)
    return result_filenames, result_labels

def augment_image(image):
    image = 255 - image
    width, height = img_size
    image = cv2.resize(image, (width, height))
    augmenter = ImageAugmenter(width, height,
                               # width and height of the image (must be the same for all images in the batch)
                               hflip=False,  # flip horizontally with 50% probability
                               vflip=False,  # flip vertically with 50% probability
                               scale_to_percent=(0.9, 1.05),  # 1.1 scale the image to 70%-130% of its original size
                               scale_axis_equally=False,  # allow the axis to be scaled unequally (e.g. x more than y)
                               rotation_deg=2,  # 2 rotate between -25 and +25 degrees
                               shear_deg=5,  # 25 shear between -10 and +10 degrees
                               translation_x_px=8,  # 1 translate between -5 and +5 px on the x-axis
                               translation_y_px=2,  # (-6, 4)
                               blur_radius=0,  # blur radius that will be applied between 0..blur_radius
                               noise_variance=0,
                               motion_blur_radius=0,
                               motion_blur_strength=0
                               )
    image = augmenter.augment_batch(np.array([image], dtype=np.uint8))[0]
    image *= 255
    image = 255 - image
    image = image.astype(np.uint8)
    # cv2.imshow("image", image)
    # cv2.waitKey(0)
    return image


def files_to_tfrecords(filenames, labels, dst_path, augment=False):
    # counter for naming .tfrecords
    current_tfrecords_index = 0

    # examples index inside .tfrecord
    current_example_index = 0

    trIdx = list(range(0, len(filenames) - 1))
    shuffle(trIdx)

    writer = tf.python_io.TFRecordWriter(os.path.join(dst_path, str(current_tfrecords_index) + ".tfrecords"))
    for example_idx in tqdm(trIdx):
        if current_example_index >= NUM_EXAMPLES_PER_RECORD:
            current_tfrecords_index += 1
            current_example_index = 0
            writer = tf.python_io.TFRecordWriter(os.path.join(dst_path, str(current_tfrecords_index) + ".tfrecords"))
            print(os.path.join(dst_path, str(current_tfrecords_index) + ".tfrecords"))
        label = labels[example_idx]
        image = cv2.imread(filenames[example_idx])
        if is_image_valid(image):
            image = cv2.resize(image, img_size)
            if augment:
                image = augment_image(image)
            else:
                # cv2.imshow("image", image)
                # cv2.waitKey(0)
                pass

            # print(label)
            # cv2.imshow("image", image)
            # cv2.waitKey(0)
            # if image.shape[2] > 1:
            #     image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # construct the Example proto object
            example = tf.train.Example(
                # Example contains a Features proto object
                features=tf.train.Features(
                    # Features contains a map of string to Feature proto objects
                    feature={
                        # A Feature contains one of either a int64_list,
                        # float_list, or bytes_list
                        'label': tf.train.Feature(
                            int64_list=tf.train.Int64List(value=label.tolist())),
                        'width': tf.train.Feature(
                            int64_list=tf.train.Int64List(value=[image.shape[0]])),
                        'height': tf.train.Feature(
                            int64_list=tf.train.Int64List(value=[image.shape[1]])),
                        'image': tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=[image.tostring()])),
                        'label_length': tf.train.Feature(
                            int64_list=tf.train.Int64List(value=[len(label)])),
                    }))
            # use the proto object to serialize the example to a string
            serialized = example.SerializeToString()
            # write the serialized object to disk
            writer.write(serialized)
            current_example_index += 1
        else:
            print("Invalid image: " + str(filenames[example_idx]))
    writer = None
    print("Finished writing tfrecords to " + dst_path)


if __name__ == '__main__':
    tfrecords_train_dst_path = os.path.join(data_dir, "tfrecords_train")
    tfrecords_test_dst_path = os.path.join(data_dir, "tfrecords_test")

    codec = load_codec()
    print(codec)
    NUM_CLASSES = get_num_classes()

    # Creating train files
    print("Reading images")
    # filenames = glob.glob(os.path.abspath(os.path.join(data_dir, "train", "*.png")))
    # filenames += filenames
    # filenames += filenames
    # filenames += filenames
    # filenames += filenames

    # shuffle(filenames)
    # filenames = filenames

    # labels = get_distilled_labels(filenames)
    # print("Writing train files")
    # files_to_tfrecords(filenames, labels, tfrecords_train_dst_path, augment=True)

    # Creating test files
    filenames = glob.glob(os.path.abspath(os.path.join(data_dir, "test", "*.png")))
    filenames += glob.glob(os.path.abspath(os.path.join(data_dir, "test", "*.jp*g")))

    labels = get_distilled_labels(filenames)
    print("Writing train files")
    files_to_tfrecords(filenames, labels, tfrecords_test_dst_path, augment=False)

    print("Finished all jobs")
