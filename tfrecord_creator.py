import os

import tensorflow as tf 
from tensorflow.keras.preprocessing import image 

# tf.enable_eager_execution()

def read_sample(serialized_example):
    feature_extraction = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
        'depth': tf.io.FixedLenFeature([], tf.int64),
    }
    example = tf.io.parse_single_example(serialized_example, feature_extraction)
    return example

def read(types="bg", tfrecord_dir=""):
    filenames = []
    for filename in os.listdir(tfrecord_dir):
        if types in filename:
            filenames.append(tfrecord_dir+filename)
    raw_dataset = tf.data.TFRecordDataset(filenames)
    parsed_dataset = raw_dataset.map(read_sample)
    return parsed_dataset

if __name__ == "__main__":
    tfrecord_dir="data/tfrecord1/"
    def _bytes_feature(value):
        """Returns a bytes_list from a string / byte."""
        # If the value is an eager tensor BytesList won't unpack a string from an EagerTensor.
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy() 
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _float_feature(value):
        """Returns a float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    def _int64_feature(value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    fg_dir = 'data/fg/'
    bg_dir = 'data/bg/'
    a_dir  = 'data/mask/'
    test_fg_dir = 'data/fg_test/'
    test_bg_dir = 'data/bg_test/'
    test_a_dir  = 'data/mask_test/'

    def serialize(image):
        shape = tf.image.decode_jpeg(image).shape
        feature = {
            'image': _bytes_feature(image),
            'height': _int64_feature(shape[0]),
            'width': _int64_feature(shape[1]),
            'depth': _int64_feature(shape[2])
        }
        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto

    # with open("./../train_names.txt") as f:
    #     names = f.read().splitlines()
    with open("data/Combined_Dataset/Training_set/training_bg_names.txt") as f:
        bg_names = f.read().splitlines()
    with open("data/Combined_Dataset/Training_set/training_fg_names.txt") as f:
        fg_names = f.read().splitlines()
    with open("data/Combined_Dataset/Test_set/test_bg_names.txt") as f:
        test_bg_names = f.read().splitlines()
    with open("data/Combined_Dataset/Test_set/test_fg_names.txt") as f:
        test_fg_names = f.read().splitlines()

    def write(type="bg"):
        if type == 'bg':
            names = bg_names
            dirr = bg_dir
            test_names = test_bg_names
            test_dirr = test_bg_dir
        elif type == 'fg':
            names = fg_names
            dirr = fg_dir
            test_names = test_fg_names
            test_dirr = test_fg_dir
        else:
            names = fg_names
            dirr = a_dir 
            test_names = test_fg_names
            test_dirr = test_a_dir

        num_images_per_shards = 2000
        num_shards = len(test_names) // num_images_per_shards + 1
        for i in range(0, len(test_names), num_images_per_shards):
            tfrecord_name = 'test_%s_%05d-of-%05d.tfrecord' % (type, i // 200, num_shards)
            with tf.io.TFRecordWriter(tfrecord_dir + tfrecord_name) as writer:
                tfrecord_sample_names = test_names[i:i+num_images_per_shards]
                for tfrecord_sample_name in tfrecord_sample_names:
                    print(test_dirr+tfrecord_sample_name)
                    image_string = open(test_dirr+tfrecord_sample_name, "rb").read()
                    # print(image_string)
                    tf_example = serialize(image_string)
                    writer.write(tf_example.SerializeToString())

        # num_images_per_shards = 2000
        # num_shards = len(names) // num_images_per_shards + 1
        # for i in range(0, len(names), num_images_per_shards):
        #     tfrecord_name = '%s_%05d-of-%05d.tfrecord' % (type, i // 200, num_shards)
        #     with tf.io.TFRecordWriter(tfrecord_dir + tfrecord_name) as writer:
        #         tfrecord_sample_names = names[i:i+num_images_per_shards]
        #         for tfrecord_sample_name in tfrecord_sample_names:
        #             print(dirr+tfrecord_sample_name)
        #             image_string = open(dirr+tfrecord_sample_name, "rb").read()
        #             # print(image_string)
        #             tf_example = serialize(image_string)
        #             writer.write(tf_example.SerializeToString())

    write("bg")
    write("a")
    write("fg")
