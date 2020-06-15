from VGG import VGG16
import tensorflow as tf


def spatial_aware(input_feature, dimension, trainable, name):
    batch, height, width, channel = input_feature.get_shape().as_list()
    vec1 = tf.reshape(tf.reduce_mean(input_feature, axis=-1), [-1, height * width])

    with tf.variable_scope(name):
        weight1 = tf.get_variable(name='weights1', shape=[height * width, int(height * width/2), dimension],
                                 trainable=trainable,
                                 initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.005),
                                 regularizer=tf.contrib.layers.l2_regularizer(0.01))
        bias1 = tf.get_variable(name='biases1', shape=[1, int(height * width/2), dimension],
                               trainable=trainable, initializer=tf.constant_initializer(0.1),
                               regularizer=tf.contrib.layers.l1_regularizer(0.01))
        # vec2 = tf.matmul(vec1, weight1) + bias1
        vec2 = tf.einsum('bi, ijd -> bjd', vec1, weight1) + bias1


        weight2 = tf.get_variable(name='weights2', shape=[int(height * width / 2), height * width, dimension],
                                  trainable=trainable,
                                  initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.005),
                                  regularizer=tf.contrib.layers.l2_regularizer(0.01))
        bias2 = tf.get_variable(name='biases2', shape=[1, height * width, dimension],
                                trainable=trainable, initializer=tf.constant_initializer(0.1),
                                regularizer=tf.contrib.layers.l1_regularizer(0.01))
        vec3 = tf.einsum('bjd, jid -> bid', vec2, weight2) + bias2

        return vec3


def SAFA(x_sat, x_grd, keep_prob, dimension, trainable):

    vgg_grd = VGG16()
    grd_local = vgg_grd.VGG16_conv(x_grd, keep_prob, trainable, 'VGG_grd')
    grd_local = tf.nn.max_pool(grd_local, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
    batch, g_height, g_width, channel = grd_local.get_shape().as_list()

    grd_w = spatial_aware(grd_local, dimension, trainable, name='spatial_grd')
    grd_local = tf.reshape(grd_local, [-1, g_height * g_width, channel])

    grd_global = tf.einsum('bic, bid -> bdc', grd_local, grd_w)
    grd_global = tf.reshape(grd_global, [-1, dimension*channel])


    vgg_sat = VGG16()
    sat_local = vgg_sat.VGG16_conv(x_sat, keep_prob, trainable, 'VGG_sat')
    sat_local = tf.nn.max_pool(sat_local, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
    batch, s_height, s_width, channel = sat_local.get_shape().as_list()

    sat_w = spatial_aware(sat_local, dimension, trainable, name='spatial_sat')
    sat_local = tf.reshape(sat_local, [-1, s_height * s_width, channel])

    sat_global = tf.einsum('bic, bid -> bdc', sat_local, sat_w)
    sat_global = tf.reshape(sat_global, [-1, dimension*channel])

    return tf.nn.l2_normalize(sat_global, dim=1), tf.nn.l2_normalize(grd_global, dim=1)




def VGG_gp(x_sat, x_grd, keep_prob, trainable):

    ############## VGG module #################

    vgg_grd = VGG16()
    grd_vgg = vgg_grd.VGG16_conv(x_grd, keep_prob, trainable, 'VGG_grd')

    vgg_sat = VGG16()
    sat_vgg = vgg_sat.VGG16_conv(x_sat, keep_prob, trainable, 'VGG_sat')

    grd_height, grd_width, grd_channel = grd_vgg.get_shape().as_list()[1:]
    grd_global = tf.nn.max_pool(grd_vgg, [1, grd_height, grd_width, 1], [1, 1, 1, 1], padding='VALID')
    grd_global = tf.reshape(grd_global, [-1, grd_channel])

    sat_height, sat_width, sat_channel = sat_vgg.get_shape().as_list()[1:]
    sat_global = tf.nn.max_pool(sat_vgg, [1, sat_height, sat_width, 1], [1, 1, 1, 1], padding='VALID')
    sat_global = tf.reshape(sat_global, [-1, sat_channel])


    return tf.nn.l2_normalize(sat_global, dim=1), tf.nn.l2_normalize(grd_global, dim=1)
