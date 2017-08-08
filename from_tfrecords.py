import tensorflow as tf
import ImageAugmenter
import cv2
from tqdm import tqdm
import glob
import os
import numpy as np
from random import shuffle


reconstructed_images = []

record_iterator = tf.python_io.tf_record_iterator(path=os.path.abspath(os.path.join("data", "train", "tfrecords", "0.tfrecords")))

for string_record in record_iterator:
    
    example = tf.train.Example()
    example.ParseFromString(string_record)
    
    height = int(example.features.feature['height']
                                 .int64_list
                                 .value[0])
    
    width = int(example.features.feature['width']
                                .int64_list
                                .value[0])
    
    img_string = (example.features.feature['image']
                                  .bytes_list
                                  .value[0])
    
    label = (example.features.feature['label']
                                .int64_list
                                .value[0])
    
    img_1d = np.fromstring(img_string, dtype=np.uint8)
    reconstructed_img = img_1d.reshape((height, width, -1))
    

    
    reconstructed_images.append(reconstructed_img)

for rec_image in tqdm(reconstructed_images):
    cv2.imshow("image", rec_image)
    cv2.waitKey(0)