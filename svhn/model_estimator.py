import h5py
import numpy as np
import tensorflow as tf

#import tftracer
#tftracer.hook_inject()

# CONVNET PARAMS
# Conv Block 1
filter_size1 = filter_size2 = 5          
num_filters1 = num_filters2 = 32        
# Conv Block 2
filter_size3 = filter_size4 = 5          
num_filters3 = num_filters4 = 64
# Conv Block 3
filter_size5 = filter_size6 = filter_size7 = 5          
num_filters5 = num_filters6 = num_filters7 = 128  
# Fully-connected layers
fc1_size = fc2_size = 256

# TRAINING PARAMS
batch_size = 512
num_epochs = 5
model_dir = "summary/dist_estimator"
benchmark_batches = 100
buffer_size = 1000

# Open the HDF5 file containing the datasets
with h5py.File('data/SVHN_multi_digit_norm_grayscale.h5','r') as h5f:
    X_train = h5f['X_train'][:]
    y_train = h5f['y_train'][:]
    X_val = h5f['X_val'][:]
    y_val = h5f['y_val'][:]
    X_test = h5f['X_test'][:]
    y_test = h5f['y_test'][:]

print('Training set', X_train.shape, y_train.shape)
print('Validation set', X_val.shape, y_val.shape)
print('Test set', X_test.shape, y_test.shape)

# Get the image data information & dimensions
train_count, img_height, img_width, num_channels = X_train.shape

# Get label information
num_digits, num_labels = y_train.shape[1], len(np.unique(y_train))

def conv_layer(inputs,    # The input or previous layer
               kernel_size,    # Width and height of each filter
               filters,    # Number of filters
               pooling,        # Average pooling
               initializer='xavier'):   # He or Xavier initialization    

    kernel_initializer = None
    if initializer == "he":
        kernel_initializer = tf.keras.initializers.he_uniform()
    else:
        kernel_initializer = tf.contrib.layers.xavier_initializer()
    
    inputs = tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=kernel_size, 
#                                use_bias=True, bias_initializer=tf.zeros_initializer(),
                                use_bias=False,
                                kernel_initializer=kernel_initializer,
                                strides=(1,1), padding="same")
    inputs = tf.layers.batch_normalization(inputs)
    inputs = tf.nn.leaky_relu(inputs, alpha=0.10)
    if pooling:
        inputs = tf.nn.avg_pool(inputs, [1,2,2,1], [1,2,2,1], 'SAME')
    return inputs

def fc_layer(inputs,  # The previous layer,         
             units,    # Num. outputs
             relu=False):    

    inputs = tf.layers.flatten(inputs)
    inputs = tf.contrib.layers.fully_connected(inputs=inputs, num_outputs=units, activation_fn=None,
                             biases_initializer=tf.zeros_initializer(),
                             weights_initializer=tf.contrib.layers.xavier_initializer())
    if relu:
        inputs = tf.nn.leaky_relu(inputs, alpha=0.10)
    return inputs


def model_fn(features, labels, mode):
    
    # Conv Block 1
    conv_1 = conv_layer(features, filter_size1, num_filters1, pooling=False)
    conv_2 = conv_layer(conv_1, filter_size2, num_filters2, pooling=True)
    drop_block1 = tf.layers.dropout(inputs=conv_2, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN ) # Dropout

    # Conv Block 2
    conv_3 = conv_layer(conv_2, filter_size3, num_filters3, pooling=False)
    conv_4 = conv_layer(conv_3, filter_size4, num_filters4, pooling=True)
    drop_block2 = tf.layers.dropout(inputs=conv_4, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN) # Dropout

    # Conv Block 3
    conv_5 = conv_layer(drop_block2, filter_size5, num_filters5, pooling=False)
    conv_6 = conv_layer(conv_5, filter_size6, num_filters6, pooling=False)
    conv_7 = conv_layer(conv_6, filter_size7, num_filters7, pooling=True)
    drop_block3 = tf.layers.dropout(inputs=conv_7, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN) # Dropout

    # Fully-connected 1
    fc_1 = fc_layer(drop_block3, fc1_size, relu=True)
    drop_fc1 = tf.layers.dropout(inputs=fc_1, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN) # Dropout

    # Fully-connected 2
    fc_2 = fc_layer(drop_fc1, fc2_size, relu=True)

    # Parallel softmax layers
    logits_1 = fc_layer(fc_2, num_labels)
    logits_2 = fc_layer(fc_2, num_labels)
    logits_3 = fc_layer(fc_2, num_labels)
    logits_4 = fc_layer(fc_2, num_labels)
    logits_5 = fc_layer(fc_2, num_labels)

    # Stack the logits together to make a prediction for an image (5 digit sequence prediction)
    logits = tf.stack([logits_1, logits_2, logits_3, logits_4, logits_5])

    predictions = {
        'classes': tf.transpose(tf.argmax(logits, axis=2)),
        'probabilities': tf.transpose(tf.nn.softmax(logits, axis=2))
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate the loss for each individual digit in the sequence
    loss1 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_1, labels=labels[:, 0]))
    loss2 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_2, labels=labels[:, 1]))
    loss3 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_3, labels=labels[:, 2]))
    loss4 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_4, labels=labels[:, 3]))
    loss5 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_5, labels=labels[:, 4]))

    # Calculate the total loss for all predictions
    loss = loss1 + loss2 + loss3 + loss4 + loss5

    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_or_create_global_step()

        # Drop learning rate by half every 20 epochs
        decay_step = 8800
        # Apply exponential decay to the learning rate
        learning_rate = tf.train.exponential_decay(1e-3, global_step, decay_step, 0.5, staircase=True)

        optimizer = tf.train.AdamOptimizer(learning_rate)

        train_op = optimizer.minimize(loss=loss, global_step=global_step)

    else:
        train_op = None

    accuracy = tf.metrics.accuracy(labels, predictions['classes'])
    metrics = {'accuracy': accuracy}

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=metrics)

def train_input_fn(input_context=None):
    dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    if input_context:
        dataset = dataset.shard(input_context.num_input_pipelines, input_context.input_pipeline_id)
    dataset = dataset.shuffle(buffer_size).repeat().batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)
    return dataset

def synth_input_fn(input_context=None):
    inputs = tf.truncated_normal(
        [img_height, img_width, num_channels],
        mean=127,
        stddev=60)
    labels = tf.random_uniform(
        [num_digits],
        minval=0,
        maxval=num_labels-1,
        dtype=tf.int32)
    dataset = tf.data.Dataset.from_tensors((inputs, labels))
    if input_context:
        dataset = dataset.shard(input_context.num_input_pipelines, input_context.input_pipeline_id)
    dataset = dataset.shuffle(buffer_size).repeat().batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)
    return dataset

def eval_input_fn(input_context=None):
    dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    if input_context:
        dataset = dataset.shard(input_context.num_input_pipelines, input_context.input_pipeline_id)
    dataset = dataset.batch(batch_size)
    return dataset

tf.logging.set_verbosity( tf.logging.INFO )

mirrored_strategy = tf.distribute.MirroredStrategy()
config = tf.estimator.RunConfig(train_distribute=mirrored_strategy, eval_distribute=mirrored_strategy,
                                save_checkpoints_steps=None, save_checkpoints_secs=None,
                                save_summary_steps=None)

classifier = tf.estimator.Estimator(model_fn=model_fn, model_dir=model_dir, config=config)

for epoch in range(num_epochs):
    classifier.train(input_fn=train_input_fn)
    classifier.evaluate(input_fn=eval_input_fn)
    classifier.evaluate(input_fn=test_input_fn)
