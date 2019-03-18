import numpy as np 
import tensorflow as tf 

def _variable_on_cpu(name, shape, initializer, use_fp16=False):
    """Helper to create a Variable stored on CPU memory.
    Args:
        name: name of the variable
        shape: list of ints
        initializer: initializer for Variable
    Returns:
        Variable Tensor
    """
    with tf.device("/cpu:0"):
        dtype = tf.float16 if use_fp16 else tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var


def _variable_with_weight_decay(name, shape, stddev, wd, use_xavier=True):
    """Helper to create an initialized Variable with weight decay.

    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.

    Args:
        name: name of the variable
        shape: list of ints
        stddev: standard deviation of a truncated Gaussian
        wd (weight decay): add L2Loss weight decay multiplied by this float.
            If None, weight decay is not added for this Variable.
        use_xavier: bool, whether to use xavier initializer

    Returns:
        Variable Tensor
    """
    if use_xavier:
        initializer = tf.contrib.layers.xavier_initializer()
    else:
        initializer = tf.truncated_normal_initializer(stddev=stddev)
    var = _variable_on_cpu(name, shape, initializer)
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def conv1d(inputs, num_output_channels, kernel_size, scope, stride=1,
           padding="SAME", data_format="NHWC", use_xavier=True,
           stddev=1e-3, weight_decay=None, activation_fn=tf.nn.relu,
           bn=False, bn_decay=None, is_training=None):
    with tf.variable_scope(scope) as sc:
        assert(data_format=="NHWC" or data_format=="NCHW")
        if data_format == "NHWC":
            num_in_channels = inputs.get_shape()[-1].value
        else:
            num_in_channels = inputs.get_shape()[1].value
        kernel_shape = [
            kernel_size, num_in_channels, num_output_channels
        ]
        kernel = _variable_with_weight_decay("weights", shape=kernel_shape,
            use_xavier=use_xavier, stddev=stddev, wd=weight_decay)
        outputs = tf.nn.conv1d(
            inputs, kernel, stride=stride, padding=padding,
            data_format=data_format
        )
        biases = _variable_on_cpu(
            "biases", [num_output_channels], tf.constant_initializer(0.0)
        )
        outputs = tf.nn.bias_add(outputs, biases, data_format=data_format)

        if activation_fn is not None:
            outputs = activation_fn(outputs)
        if bn:
            outputs = batch_norm(
                outputs, is_training, "bn", bn_decay=bn_decay,
                data_format=data_format
            )
        
        return outputs


def conv2d(inputs,
           num_output_channels,
           kernel_size,
           scope,
           stride=[1,1],
           padding="SAME",
           data_format="NHWC",
           use_xavier=True,
           stddev=1e-3,
           weight_decay=None,
           activation_fn=tf.nn.relu,
           bn=False,
           bn_decay=None,
           is_training=None):
    with tf.variable_scope(scope) as sc:
        kernel_h, kernel_w = kernel_size
        assert(data_format=="NHWC" or data_format=="NCHW")
        if data_format == "NHWC":
            num_in_channels = inputs.get_shape()[-1].value
        else:
            num_in_channels = inputs.get_shape()[1].value
        kernel_shape = [
            kernel_h, kernel_w, num_in_channels, num_output_channels
        ]
        kernel = _variable_with_weight_decay(
            "weights", shape=kernel_shape, use_xavier=use_xavier,
            stddev=stddev, wd=weight_decay
        )
        stride_h, stride_w = stride
        outputs = tf.nn.conv2d(
            inputs, kernel, [1, stride_h, stride_w, 1], padding=padding,
            data_format=data_format
        )
        biases = _variable_on_cpu(
            "biases", [num_output_channels], tf.constant_initializer(0.0)
        )
        outputs = tf.nn.bias_add(outputs, biases, data_format=data_format)

        if activation_fn is not None:
            outputs = activation_fn(outputs)
        if bn:
            outputs = batch_norm(
                outputs, is_training, "bn", bn_decay=bn_decay,
                data_format=data_format
            )
        
        return outputs


def fully_connected(inputs,
                    num_outputs,
                    scope,
                    use_xavier=True,
                    stddev=1e-3,
                    weight_decay=None,
                    activation_fn=tf.nn.relu,
                    bn=False,
                    bn_decay=None,
                    is_training=None):
    with tf.variable_scope(scope) as sc:
        num_input_units = inputs.get_shape()[-1].value
        weights = _variable_with_weight_decay(
            "weights", shape=[num_input_units, num_outputs],
            use_xavier=use_xavier, stddev=stddev, wd=weight_decay
        )
        biases = _variable_on_cpu(
            "biases", [num_outputs], tf.constant_initializer(0.0)
        )
        outputs = tf.nn.bias_add(tf.matmul(inputs, weights), biases)

        if activation_fn is not None:
            outputs = activation_fn(outputs)
        if bn:
            outputs = batch_norm(outputs, is_training, sc, bn_decay)
        
        return outputs


def max_pool2d(inputs, kernel_size, scope, stride=[1,1], padding="VALID"):
    with tf.variable_scope(scope) as sc:
        kernel_h, kernel_w = kernel_size
        stride_h, stride_w = stride
        outputs = tf.nn.max_pool(
            inputs, ksize=[1, kernel_h, kernel_w, 1],
            strides=[1, stride_h, stride_w, 1], padding=padding, name=sc.name
        )
        return outputs


def batch_norm(inputs,
               is_training,
               scope,
               bn_decay,
               data_format="NHWC"):
    with tf.variable_scope(scope) as sc:
        bn_decay = bn_decay if bn_decay is not None else 0.9
        return tf.contrib.layers.batch_norm(
            inputs, center=True, scale=True, is_training=is_training,
            decay=bn_decay, updates_collections=None,
            data_format=data_format
        )


def dropout(
    inputs,
    is_training,
    scope,
    keep_prob=0.5,
    noise_shape=None
):
    with tf.variable_scope(scope) as sc:
        outputs = tf.cond(
            is_training,
            lambda: tf.nn.dropout(inputs, keep_prob, noise_shape),
            lambda: inputs
        )
    return outputs
