import tensorflow as tf
from tensorflow.python.ops import array_ops
import numpy as np
import math
import sys
import os
import logging
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'utils'))
from utils import tf_util
from transform_nets import input_transform_net, feature_transform_net


def placeholder_inputs(batch_size, num_point, num_channels, label_type="normal"):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, num_channels))
    if label_type == "onehot":
        labels_pl = tf.placeholder(tf.int8, shape=(batch_size, 10))
    else:
        labels_pl = tf.placeholder(tf.float32, shape=(batch_size, 1))
    return pointclouds_pl, labels_pl


def get_model(point_cloud, is_training, mask=None, bn_decay=None, label_type="normal"):
    """Classification PointNet, input is BxNx3, output Bx40
        arguments:
            point_clound: numpy array
            is_training: boolean
            bn_decay: boolean or None
        output:
            tensorflow model
    """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value

    with tf.variable_scope('transform_net1') as sc:
        transform = input_transform_net(point_cloud, is_training, bn_decay, K=3)

        point_cloud_transformed = tf.matmul(point_cloud, transform)
        input_image = tf.expand_dims(point_cloud_transformed, -1)

    # Point functions (MLP implemented as conv2d)
    with tf.variable_scope("conv1") as sc:
        net_conv1 = tf_util.conv2d(input_image, 64, [1, 3],
                            padding='VALID', stride=[1,1],
                            bn=True, is_training=is_training,
                            scope='conv1_1', bn_decay=bn_decay)
        net_conv3 = tf.pad(input_image, [[0, 0], [1, 1], [0, 0], [0, 0]])
        net_conv3 = tf_util.conv2d(net_conv3, 64, [3, 3],
                            padding='VALID', stride=[1,1],
                            bn=True, is_training=is_training,
                            scope='conv1_3', bn_decay=bn_decay)
        net_conv5 = tf.pad(input_image, [[0, 0], [2, 2], [0, 0], [0, 0]])
        net_conv5 = tf_util.conv2d(net_conv5, 64, [5, 3],
                            padding='VALID', stride=[1,1],
                            bn=True, is_training=is_training,
                            scope='conv1_5', bn_decay=bn_decay)
        net = tf.concat([net_conv1, net_conv3, net_conv5], 3, name="concat_1")

    with tf.variable_scope("conv2") as sc:
        net_conv1 = tf_util.conv2d(net, 128, [1,1],
                            padding='VALID', stride=[1,1],
                            bn=True, is_training=is_training,
                            scope='conv2_1', bn_decay=bn_decay)
        net_conv3 = tf.pad(net, [[0, 0], [1, 1], [0, 0], [0, 0]])
        net_conv3 = tf_util.conv2d(net_conv3, 128, [3,1],
                            padding='VALID', stride=[1,1],
                            bn=True, is_training=is_training,
                            scope='conv2_3', bn_decay=bn_decay)
        net_conv5 = tf.pad(net, [[0, 0], [2, 2], [0, 0], [0, 0]])
        net_conv5 = tf_util.conv2d(net_conv5, 128, [5,1],
                            padding='VALID', stride=[1,1],
                            bn=True, is_training=is_training,
                            scope='conv2_5', bn_decay=bn_decay)
        net = tf.concat([net_conv1, net_conv3, net_conv5], 3, name="concat_2")

    with tf.variable_scope("conv3") as sc:
        net_conv1 = tf_util.conv2d(net, 256, [1,1],
                            padding='VALID', stride=[1,1],
                            bn=True, is_training=is_training,
                            scope='conv3_1', bn_decay=bn_decay)
        net_conv3 = tf.pad(net, [[0, 0], [1, 1], [0, 0], [0, 0]])
        net_conv3 = tf_util.conv2d(net_conv3, 256, [3,1],
                            padding='VALID', stride=[1,1],
                            bn=True, is_training=is_training,
                            scope='conv3_3', bn_decay=bn_decay)
        net_conv5 = tf.pad(net, [[0, 0], [2, 2], [0, 0], [0, 0]])
        net_conv5 = tf_util.conv2d(net_conv5, 256, [5,1],
                            padding='VALID', stride=[1,1],
                            bn=True, is_training=is_training,
                            scope='conv3_5', bn_decay=bn_decay)
        net = tf.concat([net_conv1, net_conv3, net_conv5], 3, name="concat_3")

    # masking
    net = tf.multiply(net, tf.cast(mask, tf.float32))

    net = tf_util.max_pool2d(net, kernel_size=[num_point, 1],
                            scope='max_pooling', padding='VALID')

    net = tf_util.conv2d(net, 512, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv4', bn_decay=bn_decay)
    
    # MLP on global point cloud vector
    net = tf.reshape(net, [batch_size, -1])
    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
                                  scope='fc1', bn_decay=bn_decay)
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
                                  scope='fc2', bn_decay=bn_decay)
    net = tf_util.fully_connected(net, 128, bn=True, is_training=is_training,
                                  scope='fc3', bn_decay=bn_decay)
    # net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
    #                       scope='dp1')
    if label_type == "normal":
        output = tf_util.fully_connected(net, 1, scope='fc4', activation_fn=None)
    elif label_type == "onehot":
        output = tf_util.fully_connected(net, 10, scope="fc4", activation_fn=None)
    
    return output

def get_loss(pred, label, type="l2_loss"):
    """pred: B*NUM_CLASSES,
    label: B
    """
    if type == "l2_loss":
        loss = tf.square(tf.subtract(pred, label))
        assess_loss = tf.reduce_mean(loss)
        # loss = tf.nn.l2_loss(tf.subtract(pred, label))
        # assess_loss = tf.reduce_mean(loss)
        
    elif type == "corre_loss":
        _, loss = tf.contrib.metrics.streaming_pearson_correlation(pred, label)

    elif type == "l1_loss":
        loss = tf.losses.absolute_difference(label, pred)
        assess_loss = tf.reduce_mean(loss)

    elif type == "bengio_loss":
        l_pred = pred.get_shape()[0]
        omega = tf.cast(
            tf.greater(
                tf.abs(
                    label - array_ops.expand_dims(
                        array_ops.squeeze(label),
                        0
                    )
                ),
                0.1
            ),
            tf.float32
        )
        
        y = tf.cast(
            tf.greater(
                array_ops.expand_dims(array_ops.squeeze(label), 0),
                label
            ),
            tf.float32
        )
        y = tf.subtract(tf.add(y, y), 1.0)
        
        s = tf.subtract(pred, array_ops.expand_dims(array_ops.squeeze(pred), 0))
        
        loss_mat = tf.multiply(omega, tf.maximum(0.0, tf.subtract(1.0, tf.multiply(y, s))))
        assess_loss = tf.divide(
            tf.reduce_sum(loss_mat),
            tf.cast(tf.square(l_pred), tf.float32)
        )
    elif type == "cross_entropy":
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=label, logits=pred)
        assess_loss = tf.reduce_mean(loss)

    # elif type == "val_loss":
    #     pass
    return assess_loss


if __name__ == "__main__":
    # with tf.Graph().as_default():
    #     inputs = tf.ones((32,1024,3))
    #     outputs = get_model(inputs, tf.constant(True))
    #     sess = tf.Session()
    #     sess.run(tf.global_variables_initializer())
    #     output = sess.run(outputs)
    #     print(output)

    # label = array_ops.expand_dims(tf.constant(np.arange(1, 4), dtype=tf.float32), 1)
    # pred = array_ops.expand_dims(tf.constant(np.arange(1, 4), dtype=tf.float32), 1)
    # assess, ops = get_loss(pred, label, type="bengio_loss")
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # sess = tf.Session(config=config)
    # sess.run(tf.global_variables_initializer())
    # loss = sess.run(assess)
    # omega, y, s, loss_mat = sess.run(ops)
    # print(f"""assess: {loss},
    # omega: {omega}, 
    # y: {y},
    # s: {s},
    # loss_mat: {loss_mat}""")
    pass
