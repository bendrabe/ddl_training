import argparse
import inputs
import os
import tensorflow as tf
from tensorflow.python import debug as tf_debug
from tensorflow.python.client import device_lib

_NUM_TRAIN_IMAGES=1281167
_NUM_EVAL_IMAGES=50000
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

local_batch_size = args.batch_size // args.gpus
train_steps = _NUM_TRAIN_IMAGES // args.batch_size
eval_steps = _NUM_EVAL_IMAGES // args.batch_size
decay_steps = args.num_epochs*train_steps
init_learning_rate = BASE_LR * args.batch_size / 512
input_shape = (_DEFAULT_IMAGE_SIZE, _DEFAULT_IMAGE_SIZE, _NUM_CHANNELS)

device_protos = device_lib.list_local_devices()
gpu_names = [x.name for x in device_protos if x.device_type == 'GPU']
sliced_gpu_names = gpu_names[:args.gpus]
print("running on devices: {}".format(sliced_gpu_names))
strategy = tf.distribute.MirroredStrategy(devices=sliced_gpu_names)

tf.keras.backend.set_image_data_format(DATA_FORMAT)

if args.data_dir is not None:
    input_fn = inputs.get_real_input_fn
else:
    input_fn = inputs.get_synth_input_fn(list(input_shape), _NUM_CLASSES)

train_input_dataset = input_fn(
    is_training=True,
    data_dir=args.data_dir,
    batch_size=local_batch_size,
    num_epochs=args.num_epochs)

eval_input_dataset = input_fn(
    is_training=False,
    data_dir=args.data_dir,
    batch_size=local_batch_size,
    num_epochs=1)

def fire_module(inputs, squeeze_depth, expand_depth):
    x = tf.keras.layers.Conv2D(
        filters=squeeze_depth,
        kernel_size=[1, 1],
        activation=tf.keras.activations.relu,
        kernel_initializer=tf.variance_scaling_initializer(),
        kernel_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY)) (inputs)
    e1x1 = tf.keras.layers.Conv2D(
        filters=expand_depth,
        kernel_size=[1, 1],
        activation=tf.keras.activations.relu,
        kernel_initializer=tf.variance_scaling_initializer(),
        kernel_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY)) (x)
    e3x3 = tf.keras.layers.Conv2D(
        filters=expand_depth,
        kernel_size=[3, 3],
        padding='same',
        activation=tf.keras.activations.relu,
        kernel_initializer=tf.variance_scaling_initializer(),
        kernel_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY)) (x)
    x = tf.keras.layers.Concatenate(1) ([e1x1, e3x3])
    return x

with strategy.scope():
    inputs = tf.keras.Input(shape=input_shape)

    if tf.keras.backend.image_data_format() == 'channels_first':
        x = tf.keras.layers.Lambda(
            lambda x: tf.keras.backend.permute_dimensions(x, (0, 3, 1, 2))
        ) (inputs)
    else:
        x = inputs

    x = tf.keras.layers.Conv2D(
        filters=96,
        kernel_size=[7, 7],
        strides=2,
        activation=tf.keras.activations.relu,
        kernel_initializer=tf.variance_scaling_initializer,
        kernel_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY)) (x)
    x = tf.keras.layers.MaxPool2D(
        pool_size=[3, 3],
        strides=2) (x)
    x = fire_module(x, 16, 64)
    x = fire_module(x, 16, 64)
    x = fire_module(x, 32, 128)
    x = tf.keras.layers.MaxPool2D(
        pool_size=[3, 3],
        strides=2) (x)
    x = fire_module(x, 32, 128)
    x = fire_module(x, 48, 192)
    x = fire_module(x, 48, 192)
    x = fire_module(x, 64, 256)
    x = tf.keras.layers.MaxPool2D(
        pool_size=[3, 3],
        strides=2) (x)
    x = fire_module(x, 64, 256)
    x = tf.keras.layers.Dropout(rate=0.5) (x)
    x = tf.keras.layers.Conv2D(
        filters=_NUM_CLASSES,
        kernel_size=[1, 1],
        activation=tf.keras.activations.relu,
        kernel_initializer=tf.keras.initializers.random_normal(mean=0.0, stddev=0.01),
        kernel_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY)) (x)
    x = tf.keras.layers.GlobalAveragePooling2D() (x)
    outputs = tf.keras.layers.Activation('softmax', dtype='float32')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='squeezenet')

    learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=init_learning_rate,
        decay_steps=decay_steps,
        end_learning_rate=0.0,
        power=1.0)

    optimizer = tf.keras.optimizers.SGD(
        learning_rate=learning_rate_fn,
        momentum=MOMENTUM)

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
        run_eagerly=False
    )

callbacks = []
callbacks.append(
    tf.keras.callbacks.TensorBoard(
        log_dir=args.model_dir,
        update_freq=100,
        profile_batch=0
    )
)
ckpt_full_path = os.path.join(args.model_dir, 'model.ckpt-{epoch:04d}')
callbacks.append(tf.keras.callbacks.ModelCheckpoint(ckpt_full_path, save_weights_only=True))

model.fit(
    train_input_dataset,
    epochs=args.num_epochs,
    steps_per_epoch=train_steps,
    callbacks=callbacks,
    validation_data=eval_input_dataset,
    validation_steps=eval_steps, # see github issue 28995
    validation_freq=1,
    verbose=0 # see github issue 28995, TensorBoard log anyway
)
