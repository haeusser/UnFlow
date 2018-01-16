import os
import random

import numpy as np
import tensorflow as tf

from ..core.input import Input


class EGFInput(Input):
    def __init__(self, data, batch_size, dims, *,
                 num_threads=1):
        super().__init__(data, batch_size, dims, num_threads=num_threads,
                         normalize=False)
        data_dirs = self.data.data_dir
        self.train_file = os.path.join(data_dirs, 'train.tfrecords')
        self.test_file = os.path.join(data_dirs, 'test.tfrecords')

    def _resize_crop_or_pad(self, tensor):
        raise RuntimeError("Should not be used!")

    def _resize_image_fixed(self, image):
        raise RuntimeError("Should not be used!")

    def _normalize_image(self, image):
        raise RuntimeError("Should not be used!")

    def _preprocess_image(self, image):
        raise RuntimeError("Should not be used!")

    def _input_images(self, image_dir, hold_out_inv=None):
        """Assumes that paired images are next to each other after ordering the
        files.
        """
        raise NotImplementedError

    def _input_test(self, image_dir, hold_out_inv=None):
        raise RuntimeError("Should not be used!")

    def get_normalization(self):
        return 0., 1.

    def _get_input(self, split='train'):
        height, width = self.dims

        if split == 'train':
            input_file = self.train_file
            scope_name = 'train_inputs'
            testing = False
        else:
            input_file = self.test_file
            scope_name = 'test_inputs'
            testing = True

        with tf.variable_scope(scope_name):

            feature = {
                'identifier': tf.FixedLenFeature([], tf.string),
                'electrode_1': tf.FixedLenFeature([], tf.string),
                'electrode_2': tf.FixedLenFeature([], tf.string),
                'flow': tf.FixedLenFeature([], tf.string)
            }

            filename_queue = tf.train.string_input_producer([input_file])

            reader = tf.TFRecordReader()
            _, serialized_example = reader.read(filename_queue)
            features = tf.parse_single_example(serialized_example,
                                               features=feature)

            # identifier = features['identifier']
            electrode_1 = tf.decode_raw(features['electrode_1'], tf.uint8)
            electrode_2 = tf.decode_raw(features['electrode_2'], tf.uint8)

            electrode_1.set_shape([64])
            electrode_2.set_shape([64])

            electrode_1 = tf.reshape(electrode_1, [8, 8])
            electrode_2 = tf.reshape(electrode_2, [8, 8])

            electrode_1_up = tf.image.resize_bicubic(
                [tf.expand_dims(electrode_1, -1)], [height, width])
            electrode_2_up = tf.image.resize_bicubic(
                [tf.expand_dims(electrode_2, -1)], [height, width])

            electrode_1_up = tf.squeeze(electrode_1_up)
            electrode_2_up = tf.squeeze(electrode_2_up)
            electrode_1_up = tf.expand_dims(electrode_1_up, -1)
            electrode_2_up = tf.expand_dims(electrode_2_up, -1)

            # Hack
            electrode_1_up = tf.image.grayscale_to_rgb(electrode_1_up)
            electrode_2_up = tf.image.grayscale_to_rgb(electrode_2_up)

            electrode_1_up.set_shape([height, width, 3])
            electrode_2_up.set_shape([height, width, 3])

            fetches = [electrode_1_up, electrode_2_up]


            if testing:
                flow = tf.decode_raw(features['flow'],
                                     tf.float32)
                flow.set_shape([200 * 200 * 2])
                flow = tf.reshape(flow, [200, 200, 2])
                fetches.append(flow)

            return tf.train.batch(
                fetches,
                batch_size=self.batch_size,
                num_threads=self.num_threads)
            #return tf.train.shuffle_batch(fetches,
            #                              batch_size=self.batch_size,
            #                              num_threads=self.num_threads,
            #                              capacity=3*self.batch_size,
            #                              min_after_dequeue=10)

    def input_raw(self, swap_images=True, sequence=True,
                  needs_crop=True, shift=0, seed=0,
                  center_crop=False, skip=0):
        """Constructs input of raw data.

        Args:
        Returns:
            image_1: batch of first images
            image_2: batch of second images
        """
        return self._get_input()


    def input_gt(self, image_dir=None, flow_dir=None, hold_out_inv=None):
        return self._get_input(split='test')