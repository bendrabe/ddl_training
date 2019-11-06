import argparse
import os
import tensorflow as tf
from tensorflow.layers import conv2d, average_pooling2d, max_pooling2d, dropout
#from tensorflow.python import debug as tf_debug

import inputs
import squeezenet

_NUM_TRAIN_IMAGES=1281167
_DEFAULT_IMAGE_SIZE = 227
_NUM_CHANNELS = 3
_NUM_CLASSES = 1000
DATA_FORMAT='channels_first'
BASE_LR=0.04
WEIGHT_DECAY=0.0002
MOMENTUM=0.9

parser = argparse.ArgumentParser()
parser.add_argument("-g", "--gpus", type=int, help="GPUs per node", default=4, choices=[1,2,4])
parser.add_argument("-b", "--batch_size", type=int, help="global batch size", default=512)
parser.add_argument("-ne", "--num_epochs", type=int, help="number of epochs to train", default=70)
parser.add_argument("-dd", "--data_dir", type=str, help="path to ImageNet data", default=None)
parser.add_argument("-md", "--model_dir", type=str, help="path to store summaries / checkpoints", default='summary')
args = parser.parse_args()

def model_fn(features, labels, mode, params):
    tf.summary.image("inputs", tf.transpose(features, [0,2,3,1]), max_outputs=6)
    lr0 = params['lr0']
    decay_steps = params['decay_steps']
    weight_decay = params['weight_decay']
    data_format = params['data_format']
    is_training = mode == tf.estimator.ModeKeys.TRAIN

    logits = squeezenet.model(features, is_training, data_format, _NUM_CLASSES)

    predictions = {
        'classes': tf.argmax(logits, axis=1),
        'probabilities': tf.nn.softmax(logits, axis=1)
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    cross_entropy = tf.losses.sparse_softmax_cross_entropy(
        logits=logits, labels=labels
    )
    tf.identity(cross_entropy, name='cross_entropy')
    tf.summary.scalar('cross_entropy', cross_entropy)

    l2_loss = weight_decay * tf.add_n(
        [tf.nn.l2_loss(v) for v in tf.trainable_variables()]
    )
    tf.identity(l2_loss, name='l2_loss')
    tf.summary.scalar('l2_loss', l2_loss)

    loss = cross_entropy + l2_loss

    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_or_create_global_step()
        learning_rate = tf.train.polynomial_decay(
            learning_rate=lr0,
            global_step=global_step,
            decay_steps=decay_steps,
            end_learning_rate=0.0,
            power=1.0
        )
        tf.identity(learning_rate, 'learning_rate')
        tf.summary.scalar('learning_rate', learning_rate)
        optimizer = tf.train.MomentumOptimizer(
            learning_rate=learning_rate, momentum=MOMENTUM
        )
        grad_vars = optimizer.compute_gradients(loss)
        minimize_op = optimizer.apply_gradients(grad_vars, global_step)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        train_op = tf.group(minimize_op, update_ops)
    else:
        train_op = None

    accuracy = tf.metrics.accuracy(labels, predictions['classes'])
    accuracy_top_5 = tf.metrics.mean(
        tf.nn.in_top_k(
            predictions=logits, targets=labels, k=5, name='top_5_op'
        )
    )

    metrics = {
        'accuracy': accuracy,
        'accuracy_top_5': accuracy_top_5
    }

    # Create a tensor named train_accuracy for logging purposes
    tf.identity(accuracy[1], name='train_accuracy')
    tf.identity(accuracy_top_5[1], name='train_accuracy_top_5')
    tf.summary.scalar('train_accuracy', accuracy[1])
    tf.summary.scalar('train_accuracy_top_5', accuracy_top_5[1])

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=metrics)

local_batch_size = args.batch_size // args.gpus
steps_per_epoch = ((_NUM_TRAIN_IMAGES - 1 ) // args.batch_size) + 1
decay_steps = args.num_epochs*steps_per_epoch
lr0 = BASE_LR * args.batch_size / 512
input_shape = (_DEFAULT_IMAGE_SIZE, _DEFAULT_IMAGE_SIZE, _NUM_CHANNELS)

if args.data_dir is not None:
    input_fn = inputs.get_real_input_fn
else:
    input_fn = inputs.get_synth_input_fn(list(input_shape), _NUM_CLASSES)
    lr0 = 0.0 # protect from NaN

def train_input_fn():
    return input_fn(
        is_training=True,
        data_dir=args.data_dir,
        batch_size=local_batch_size
    )

def eval_input_fn():
    return input_fn(
        is_training=False,
        data_dir=args.data_dir,
        batch_size=local_batch_size
    )

tf.logging.set_verbosity( tf.logging.INFO )

session_config = tf.ConfigProto(allow_soft_placement=True)

config = tf.estimator.RunConfig(
    session_config=session_config,
    save_checkpoints_steps=steps_per_epoch,
    train_distribute=tf.contrib.distribute.MirroredStrategy(num_gpus=args.gpus)
)

classifier = tf.estimator.Estimator(model_fn=model_fn, model_dir=args.model_dir,
    config=config, params={
        'lr0': lr0,
        'decay_steps': decay_steps,
        'weight_decay': WEIGHT_DECAY,
        'data_format': DATA_FORMAT
    })

_TENSORS_TO_LOG = dict((x, x) for x in ['learning_rate',
                                        'cross_entropy',
                                        'train_accuracy'])

hooks = [tf.train.LoggingTensorHook(_TENSORS_TO_LOG, every_n_iter=100)]
#hooks = [tf_debug.LocalCLIDebugHook()]
#hooks = [tf.train.ProfilerHook(save_steps=1000)]

for _ in range(args.num_epochs):
    classifier.train(input_fn=train_input_fn, steps=steps_per_epoch, hooks=hooks)
    classifier.evaluate(input_fn=eval_input_fn)
