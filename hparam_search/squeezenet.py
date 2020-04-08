import tensorflow as tf
from tensorflow.layers import conv2d, average_pooling2d, max_pooling2d, dropout

def fire_module(inputs, squeeze_depth, expand_depth, data_format):
    net = _squeeze(inputs, squeeze_depth, data_format)
    net = _expand(net, expand_depth, data_format)
    return net

def _squeeze(inputs, num_outputs, data_format):
    return conv2d(inputs=inputs,
                  filters=num_outputs,
                  kernel_size=[1, 1],
                  strides=1,
                  padding='valid',
                  data_format=data_format,
                  activation=tf.nn.relu,
                  use_bias=True,
                  kernel_initializer=tf.variance_scaling_initializer(),
                  bias_initializer=tf.zeros_initializer())

def _expand(inputs, num_outputs, data_format):
    e1x1 = conv2d(inputs=inputs,
                  filters=num_outputs,
                  kernel_size=[1, 1],
                  strides=1,
                  padding='valid',
                  data_format=data_format,
                  activation=tf.nn.relu,
                  use_bias=True,
                  kernel_initializer=tf.variance_scaling_initializer(),
                  bias_initializer=tf.zeros_initializer())

    e3x3 = conv2d(inputs=inputs,
                  filters=num_outputs,
                  kernel_size=[3, 3],
                  strides=1,
                  padding='same',
                  data_format=data_format,
                  activation=tf.nn.relu,
                  use_bias=True,
                  kernel_initializer=tf.variance_scaling_initializer(),
                  bias_initializer=tf.zeros_initializer())

    return tf.concat([e1x1, e3x3], 1)

def model(inputs, is_training, data_format, num_classes):
    net = conv2d(inputs=inputs,
                 filters=96,
                 kernel_size=[7, 7],
                 strides=2,
                 padding='valid',
                 data_format=data_format,
                 activation=tf.nn.relu,
                 use_bias=True,
                 kernel_initializer=tf.variance_scaling_initializer(),
                 bias_initializer=tf.zeros_initializer())
    net = max_pooling2d(inputs=net,
                        pool_size=[3, 3],
                        strides=2,
                        data_format=data_format)
    net = fire_module(net, 16, 64, data_format)
    net = fire_module(net, 16, 64, data_format)
    net = fire_module(net, 32, 128, data_format)
    net = max_pooling2d(inputs=net,
                     pool_size=[3, 3],
                     strides=2,
                     data_format=data_format)
    net = fire_module(net, 32, 128, data_format)
    net = fire_module(net, 48, 192, data_format)
    net = fire_module(net, 48, 192, data_format)
    net = fire_module(net, 64, 256, data_format)
    net = max_pooling2d(inputs=net,
                     pool_size=[3, 3],
                     strides=2,
                     data_format=data_format)
    net = fire_module(net, 64, 256, data_format)
    net = dropout(inputs=net,
                  rate=0.5,
                  training=is_training)
    net = conv2d(inputs=net,
                 filters=num_classes,
                 kernel_size=[1, 1],
                 strides=1, 
                 padding='valid', # no padding eqv. to pad=1 for 1x1 conv?
                 data_format=data_format,
                 activation=tf.nn.relu,
                 use_bias=True,
                 kernel_initializer=tf.initializers.random_normal(mean=0.0, stddev=0.01),
                 bias_initializer=tf.zeros_initializer())
    net = average_pooling2d(inputs=net,
                            pool_size=[13, 13],
                            strides=1,
                            data_format=data_format)

    # TODO fix for data_format later
    logits = tf.squeeze(net, [2,3])

    return logits
