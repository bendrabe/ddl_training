import argparse
import json
import numpy as np
import tensorflow as tf
#from tensorflow.python import debug as tf_debug

import inputs
import squeezenet

_NUM_TRAIN_IMAGES=1281167
_DEFAULT_IMAGE_SIZE = 227
_NUM_CHANNELS = 3
_NUM_CLASSES = 1000
DATA_FORMAT='channels_first'
MOMENTUM=0.9

def mix(batch_size, alpha, images, labels):
    """Applies Mixup regularization to a batch of images and labels.
    
    [1] Hongyi Zhang, Moustapha Cisse, Yann N. Dauphin, David Lopez-Paz
        Mixup: Beyond Empirical Risk Minimization.
        ICLR'18, https://arxiv.org/abs/1710.09412
    
    Arguments:
        batch_size: The input batch size for images and labels.
        alpha: Float that controls the strength of Mixup regularization.
        images: A batch of images of shape [batch_size, ...]
        labels: A batch of labels of shape [batch_size, num_classes]
    
    Returns:
        A tuple of (images, labels) with the same dimensions as the input with
        Mixup regularization applied.
    """
    mix_weight = tf.distributions.Beta(alpha, alpha).sample([batch_size, 1])
    mix_weight = tf.maximum(mix_weight, 1. - mix_weight)
    images_mix_weight = tf.reshape(mix_weight, [batch_size, 1, 1, 1])
    # Mixup on a single batch is implemented by taking a weighted sum with the
    # same batch in reverse.
    images_mix = (
        images * images_mix_weight + images[::-1] * (1. - images_mix_weight))
    labels_mix = labels * mix_weight + labels[::-1] * (1. - mix_weight)
    return images_mix, labels_mix

def model_fn(features, labels, mode, params):
    tf.summary.image("inputs", tf.transpose(features, [0,2,3,1]), max_outputs=6)
    mixup = params['mixup']
    local_batch_size = params['local_batch_size']
    lr0 = params['lr0']
    lr_decay_sched = params['lr_decay_sched']
    lr_decay_rate = params['lr_decay_rate']
    warmup_steps = params['warmup_steps']
    decay_steps = params['decay_steps']
    weight_decay = params['weight_decay']
    data_format = params['data_format']
    is_training = mode == tf.estimator.ModeKeys.TRAIN

    if is_training and mixup:
        features, labels = mix(local_batch_size, 0.2, features, labels)

    logits = squeezenet.model(features, is_training, data_format, _NUM_CLASSES)

    predictions = {
        'classes': tf.argmax(logits, axis=1),
        'probabilities': tf.nn.softmax(logits, axis=1)
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    cross_entropy = tf.losses.softmax_cross_entropy(
        logits=logits, onehot_labels=labels
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

        if lr_decay_sched == "exp":
            lr = tf.train.natural_exp_decay(
                learning_rate=lr0,
                global_step=global_step,
                decay_steps=1,
                decay_rate=lr_decay_rate
            )
            post_warmup_lr = lr0 * np.exp(-1 * lr_decay_rate * warmup_steps)
        else:
            lr = tf.train.polynomial_decay(
                learning_rate=lr0,
                global_step=global_step,
                decay_steps=decay_steps,
                end_learning_rate=0.0,
                power=lr_decay_rate
            )
            post_warmup_lr = lr0 * np.power(1 - warmup_steps/decay_steps, lr_decay_rate)

        if warmup_steps > 0:
            warmup_lr = (
                            tf.cast(post_warmup_lr, lr.dtype) * 
                            tf.cast(global_step, lr.dtype) / 
                            tf.cast(warmup_steps, lr.dtype)
                        )
            learning_rate = tf.cond(global_step < warmup_steps, lambda: warmup_lr, lambda: lr)
        else:
            learning_rate = lr

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

    non_onehot_labels = tf.argmax(labels, axis=1)
    accuracy = tf.metrics.accuracy(non_onehot_labels, predictions['classes'])
    accuracy_top_5 = tf.metrics.mean(
        tf.nn.in_top_k(
            predictions=logits, targets=non_onehot_labels, k=5, name='top_5_op'
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

class Experiment:
    def __init__(self,
                 test_only=False,
                 num_gpus=4,
                 num_epochs=68,
                 data_dir='/home/shared/imagenet/tfrecord/',
                 test_data_dir='/home/brabe2/imagenet-v2/imagenetv2-matched-frequency/',
                 model_dir='summary',
                 global_batch_size=512,
                 crop='squeeze',
                 std=False,
                 mixup=False,
                 lr0=0.04,
                 lr_decay_sched="poly",
                 lr_decay_rate=1.0,
                 weight_decay=0.0002,
                 warmup_epochs=0):
        self.test_only = test_only
        self.num_gpus = num_gpus
        self.num_epochs = num_epochs
        self.data_dir = data_dir
        self.test_data_dir = test_data_dir
        self.model_dir = model_dir
        self.global_batch_size = global_batch_size
        self.crop = crop
        self.std = std
        self.mixup = mixup
        self.lr0 = lr0
        self.lr_decay_sched = lr_decay_sched
        self.lr_decay_rate = lr_decay_rate
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs

        self.local_batch_size = global_batch_size // num_gpus
        self.steps_per_epoch = ((_NUM_TRAIN_IMAGES - 1 ) // global_batch_size) + 1
        self.input_shape = (_DEFAULT_IMAGE_SIZE, _DEFAULT_IMAGE_SIZE, _NUM_CHANNELS)
        self.warmup_steps = int(self.steps_per_epoch * self.warmup_epochs)
        self.decay_steps = int(self.steps_per_epoch * num_epochs)

        # TODO: error handling, make sure gbs is multiple of num_gpus
        if self.lr_decay_sched not in ["poly", "exp"]:
            raise ValueError

        self.hyperparams = {
            'global_batch_size': self.global_batch_size,
            'crop': self.crop,
            'std': self.std,
            'mixup': self.mixup,
            'lr0': self.lr0,
            'lr_decay_sched': self.lr_decay_sched,
            'lr_decay_rate': self.lr_decay_rate,
            'weight_decay': self.weight_decay,
            'warmup_epochs': self.warmup_epochs
        }

    def log_hyperparams(self):
        with open(self.model_dir + 'hyperparams.txt', 'w') as f:
            json.dump(self.hyperparams, f)

    def execute(self):
        if self.data_dir is not None:
            input_fn = inputs.get_real_input_fn
        else:
            input_fn = inputs.get_synth_input_fn(list(self.input_shape), _NUM_CLASSES)
            self.lr0 = 0.0 # protect from NaN

        def train_input_fn():
            return input_fn(
                is_training=True,
                data_dir=self.data_dir,
                batch_size=self.local_batch_size,
                crop=self.crop,
                std=self.std
            )

        def eval_input_fn():
            return input_fn(
                is_training=False,
                data_dir=self.data_dir,
                batch_size=self.local_batch_size,
                crop=self.crop,
                std=self.std
            )

        tf.logging.set_verbosity( tf.logging.INFO )

        session_config = tf.ConfigProto(allow_soft_placement=True)

        config = tf.estimator.RunConfig(
            session_config=session_config,
            save_checkpoints_steps=self.steps_per_epoch,
            keep_checkpoint_max=1,
            save_summary_steps=500,
            train_distribute=tf.contrib.distribute.MirroredStrategy(num_gpus=self.num_gpus)
        )

        classifier = tf.estimator.Estimator(model_fn=model_fn, model_dir=self.model_dir,
            config=config, params={
                'mixup': self.mixup,
                'local_batch_size': self.local_batch_size,
                'lr0': self.lr0,
                'lr_decay_sched': self.lr_decay_sched,
                'lr_decay_rate': self.lr_decay_rate,
                'weight_decay': self.weight_decay,
                'warmup_steps': self.warmup_steps,
                'decay_steps': self.decay_steps,
                'data_format': DATA_FORMAT
            })

        _TENSORS_TO_LOG = dict((x, x) for x in ['learning_rate',
                                                'cross_entropy',
                                                'train_accuracy'])

        hooks = [tf.train.LoggingTensorHook(_TENSORS_TO_LOG, every_n_iter=500)]
        #hooks = [tf_debug.LocalCLIDebugHook()]
        #hooks = [tf.train.ProfilerHook(save_steps=1000)]

        max_val = 0.0
        if not self.test_only:
            max_epoch = 0

            for epoch in range(self.num_epochs):
                classifier.train(input_fn=train_input_fn, steps=self.steps_per_epoch, hooks=hooks)
                eval_results = classifier.evaluate(input_fn=eval_input_fn, name='val')
                val = eval_results['accuracy']
                if val > max_val:
                    max_val = val
                    max_epoch = epoch
                # if peak performance was more than 10 epochs ago, quit
                if epoch - max_epoch >= 10:
                    break
            print("peak val accuracy: {}".format(max_val))

        def test_input_fn():
            return inputs.get_imagenet2_inputfn(
                data_dir=self.test_data_dir,
                batch_size=self.local_batch_size,
                std=self.std
            )
        test_results = classifier.evaluate(input_fn=test_input_fn, name='test')
        print("test accuracy: {}".format(test_results['accuracy']))

        if not self.test_only:
            val_test = {
                'val': float(max_val),
                'test': float(test_results['accuracy'])
            }

            with open(self.model_dir + 'val_test.txt', 'w') as f:
                json.dump(val_test, f)
