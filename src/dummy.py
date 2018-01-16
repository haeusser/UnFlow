import os
import numpy as np
import sys
import time
import tensorflow as tf

from e2eflow.core import flownet as fn

batch_size = 128
img_size = 256
im1 = np.zeros((batch_size, img_size, img_size, 1), dtype=np.float32)
im2 = np.zeros_like(im1)

with tf.Graph().as_default():
  f = fn.flownet(im1, im2, flownet_spec='S')
  sess_config = tf.ConfigProto(allow_soft_placement=True)
  sess_config.gpu_options.allow_growth = True
  with tf.Session(config=sess_config) as sess:

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    print('starting')
    tic = time.time()
    for _ in range(100):
      res = sess.run(f)
    toc = time.time()
    sec_per_pass = (toc-tic)/100
    sec_per_pair = sec_per_pass / batch_size
    print('timing: forward pass = %f seconds per batch at bs %d --> %f sec per pair' % (sec_per_pass, batch_size, sec_per_pair))
    coord.request_stop()
    coord.join(threads)

