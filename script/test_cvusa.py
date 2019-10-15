# import os

# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from spatial_net import *
from input_data_cvusa import InputData
import tensorflow as tf
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='TensorFlow implementation.')

parser.add_argument('--network_type',              type=str,   help='network type',      default='SAFA_8')
parser.add_argument('--polar',                     type=int,   help='polar',             default=1)

args = parser.parse_args()

# --------------  configuration parameters  -------------- #
network_type = args.network_type
polar = args.polar

data_type = 'CVUSA'

batch_size = 32
is_training = True
loss_weight = 10.0
number_of_epoch = 100

learning_rate_val = 1e-5
keep_prob_val = 0.8
# -------------------------------------------------------- #



def validate(dist_array, top_k):
    accuracy = 0.0
    data_amount = 0.0
    for i in range(dist_array.shape[0]):
        gt_dist = dist_array[i, i]
        prediction = np.sum(dist_array[:, i] < gt_dist)
        if prediction < top_k:
            accuracy += 1.0
        data_amount += 1.0
    accuracy /= data_amount
    return accuracy



if __name__ == '__main__':

    tf.reset_default_graph()
    # import data
    input_data = InputData(polar)

    # define placeholders
    if polar:
        sat_x = tf.placeholder(tf.float32, [None, 112, 616, 3], name='sat_x')
    else:
        sat_x = tf.placeholder(tf.float32, [None, 256, 256, 3], name='sat_x')

    grd_x = tf.placeholder(tf.float32, [None, 112, 616, 3], name='grd_x')

    keep_prob = tf.placeholder(tf.float32)
    learning_rate = tf.placeholder(tf.float32)


    # build model
    dimension = int(network_type[-1])
    sat_global, grd_global = SAFA(sat_x, grd_x, keep_prob, dimension, is_training)

    out_channel = sat_global.get_shape().as_list()[-1]
    sat_global_descriptor = np.zeros([input_data.get_test_dataset_size(), out_channel])
    grd_global_descriptor = np.zeros([input_data.get_test_dataset_size(), out_channel])

    print('setting saver...')
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)
    print('setting saver done...')

    # run model
    print('run model...')
    config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    print('open session ...')
    with tf.Session(config=config) as sess:
        print('initialize...')
        sess.run(tf.global_variables_initializer())

        print('load model...')

        load_model_path = '../Model/CVUSA/Trained/model.ckpt'
        saver.restore(sess, load_model_path)

        print('validate...')
        print('   compute global descriptors')
        input_data.reset_scan()

        val_i = 0
        while True:
            print('      progress %d' % val_i)
            batch_sat, batch_grd = input_data.next_batch_scan(batch_size)
            if batch_sat is None:
                break
            feed_dict = {sat_x: batch_sat, grd_x: batch_grd, keep_prob: 1.0}
            sat_global_val, grd_global_val = \
                sess.run([sat_global, grd_global], feed_dict=feed_dict)

            sat_global_descriptor[val_i: val_i + sat_global_val.shape[0], :] = sat_global_val
            grd_global_descriptor[val_i: val_i + grd_global_val.shape[0], :] = grd_global_val
            val_i += sat_global_val.shape[0]

        print('   compute accuracy')
        dist_array = 2 - 2 * np.matmul(sat_global_descriptor, np.transpose(grd_global_descriptor))
        top1_percent = int(dist_array.shape[0] * 0.01) + 1
        val_accuracy = np.zeros((1, top1_percent))
        print('start')
        for i in range(top1_percent):
            val_accuracy[0, i] = validate(dist_array, i)

        print(network_type, ':')
        print('top1', ':', val_accuracy[0, 1])
        print('top5', ':', val_accuracy[0, 5])
        print('top10', ':', val_accuracy[0, 10])
        print('top1%', ':', val_accuracy[0, -1])

