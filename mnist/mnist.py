import dataset
import numpy as np
import os
import random
import tensorflow as tf

ngpus = 4
winit_GOLD = tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG', uniform=True)
hu_GOLD = 95
act_GOLD = tf.nn.tanh
gbs_GOLD = 20
lbs_GOLD = gbs_GOLD // ngpus
lr_GOLD = 0.3109187682539478
l2_GOLD = 2.4342163691232212e-05
acc_GOLD = 0.953

min_epochs = 100
max_epochs = 1000
out_dir = "summary/mnist_GOLD_trace"
data_dir = "data"

def nn_model_fn(features, labels, mode, params):
    input_layer = tf.reshape( features, [-1, 28*28] )
    dense = tf.layers.dense(inputs=input_layer,
                            units=params['hu'],
                            activation=params['act'],
                            kernel_initializer=params['winit'])
    logits = tf.layers.dense(inputs=dense,
                             units=10,
                             kernel_initializer=tf.zeros_initializer())

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    l2_loss = params['l2'] * tf.add_n(
        [tf.nn.l2_loss(v) for v in tf.trainable_variables()]
    )
    loss = cross_entropy + l2_loss

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=params['lr'])
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])
    }
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

tf.logging.set_verbosity(tf.logging.INFO)

# load into train / test
((x_train, y_train), (x_test, y_test)) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype(np.float32)
x_train /= np.float32(255)
y_train = y_train.astype(np.int32)  # not required
x_test = x_test.astype(np.float32)
x_test /= np.float32(255)
y_test = y_test.astype(np.int32)  # not required

train_input_fn = tf.estimator.inputs.numpy_input_fn(
                     x=x_train,
                     y=y_train,
                     batch_size=lbs_GOLD,
                     num_epochs=1,
                     shuffle=True)

eval_input_fn = tf.estimator.inputs.numpy_input_fn(
                    x=x_test,
                    y=y_test,
                    num_epochs=1,
                    shuffle=False)

hyperparams = {'winit': winit_GOLD, 'hu': hu_GOLD, 'act': act_GOLD, 'bs': gbs_GOLD, 'lr': lr_GOLD, 'l2': l2_GOLD}

session_config = tf.ConfigProto(allow_soft_placement=True)

config = tf.estimator.RunConfig(
        session_config=session_config,
        train_distribute=tf.contrib.distribute.MirroredStrategy(num_gpus=ngpus)
)

mnist_classifier = tf.estimator.Estimator(model_fn=nn_model_fn, model_dir=out_dir, config=config, params=hyperparams)

max_val = 0.0
max_epoch = 0
epoch = 0
done = False

while not done:
    mnist_classifier.train(input_fn=train_input_fn)

    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    val = eval_results['accuracy']

    if val > max_val:
        max_val = val
        max_epoch = epoch

    if (epoch > max_epochs) or (epoch > min_epochs and val < max_val and max_epoch < 3*epoch/4):
        done = True

    epoch += 1
