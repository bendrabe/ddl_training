import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib

ngpus = 4
min_epochs = 5
max_epochs = 100
out_dir = "summary/mnist_tfkeras_g4"

winit_GOLD = tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG', uniform=True)
hu_GOLD = 95
act_GOLD = tf.nn.tanh
gbs_GOLD = 20
lbs_GOLD = gbs_GOLD // ngpus
lr_GOLD = 0.3109187682539478
l2_GOLD = 2.4342163691232212e-05

tf.logging.set_verbosity(tf.logging.INFO)

# load into train / test
((x_train, y_train), (x_test, y_test)) = tf.keras.datasets.mnist.load_data()

x_train = x_train.astype(np.float32)
x_train /= np.float32(255)
y_train = y_train.astype(np.int32)  # not required

print("x_train.shape = {}".format(x_train.shape))
print("y_train.shape = {}".format(y_train.shape))

x_test = x_test.astype(np.float32)
x_test /= np.float32(255)
y_test = y_test.astype(np.int32)  # not required

print("x_test.shape = {}".format(x_test.shape))
print("y_test.shape = {}".format(y_test.shape))

device_protos = device_lib.list_local_devices()
gpu_names = [x.name for x in device_protos if x.device_type == 'GPU']
sliced_gpu_names = gpu_names[:ngpus]
print("running on devices: {}".format(sliced_gpu_names))
strategy = tf.distribute.MirroredStrategy(devices=sliced_gpu_names)

with strategy.scope():
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28,28)),
        tf.keras.layers.Dense(hu_GOLD,
                           activation=act_GOLD,
                           kernel_initializer=winit_GOLD,
                           kernel_regularizer=tf.keras.regularizers.l2(l2_GOLD)),
        tf.keras.layers.Dense(10,
                           activation='softmax',
                           kernel_initializer=tf.zeros_initializer(),
                           kernel_regularizer=tf.keras.regularizers.l2(l2_GOLD))
    ])
            
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr_GOLD)

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
        run_eagerly=False
    )

callbacks = []
callbacks.append(
    tf.keras.callbacks.TensorBoard(
        log_dir=out_dir,
        update_freq=100,
        profile_batch=0
    )
)
callbacks.append(
    tf.keras.callbacks.EarlyStopping(
        monitor='val_acc',
        patience=5
    )
)

model.fit(x_train,
          y_train,
          batch_size=gbs_GOLD,
          epochs=max_epochs,
          callbacks=callbacks,
          validation_data=(x_test, y_test)
)
