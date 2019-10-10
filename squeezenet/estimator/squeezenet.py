import os
import tensorflow as tf
from tensorflow.layers import conv2d, average_pooling2d, max_pooling2d, dropout

#from tensorflow.python import debug as tf_debug

_NUM_TRAIN_FILES=1024
_NUM_TRAIN_IMAGES=1281167
_SHUFFLE_BUFFER=10000
_DEFAULT_IMAGE_SIZE = 227
_NUM_CHANNELS = 3
_NUM_CLASSES = 1000

DATA_FORMAT='channels_first'
NUM_GPUS=4
G_BATCH_SIZE=512
BASE_LR=0.04
WEIGHT_DECAY=0.0002
MOMENTUM=0.9
NUM_EPOCHS=68
DATA_DIR='/data/imagenet-tfrecord/'
MODEL_DIR='summary/ep68_g4'

L_BATCH_SIZE = G_BATCH_SIZE//NUM_GPUS
# num_epochs times (ceil division of train images by global batch size) 
MAX_STEPS = NUM_EPOCHS*(((_NUM_TRAIN_IMAGES - 1 ) // G_BATCH_SIZE) + 1)

def fire_module(inputs, squeeze_depth, expand_depth, weight_decay,
				data_format):
	net = _squeeze(inputs, squeeze_depth, weight_decay, data_format)
	net = _expand(net, expand_depth, weight_decay, data_format)
	return net

def _squeeze(inputs, num_outputs, weight_decay, data_format):
	return conv2d(inputs=inputs,
				  filters=num_outputs,
				  kernel_size=[1, 1],
				  strides=1,
				  padding='valid',
				  data_format=data_format,
				  activation=tf.nn.relu,
				  use_bias=True,
				  kernel_initializer=tf.variance_scaling_initializer(),
				  bias_initializer=tf.zeros_initializer(),
				  kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay))

def _expand(inputs, num_outputs, weight_decay, data_format):
	e1x1 = conv2d(inputs=inputs,
				  filters=num_outputs,
				  kernel_size=[1, 1],
				  strides=1,
				  padding='valid',
				  data_format=data_format,
				  activation=tf.nn.relu,
				  use_bias=True,
				  kernel_initializer=tf.variance_scaling_initializer(),
				  bias_initializer=tf.zeros_initializer(),
				  kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay))

	e3x3 = conv2d(inputs=inputs,
				  filters=num_outputs,
				  kernel_size=[3, 3],
				  strides=1,
				  padding='same',
				  data_format=data_format,
				  activation=tf.nn.relu,
				  use_bias=True,
				  kernel_initializer=tf.variance_scaling_initializer(),
				  bias_initializer=tf.zeros_initializer(),
				  kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay))

	return tf.concat([e1x1, e3x3], 1)

def model_fn(features, labels, mode, params):
	wd = params['weight_decay']
	df = params['data_format']
	net = conv2d(inputs=features,
				 filters=96,
				 kernel_size=[7, 7],
				 strides=2,
				 padding='valid',
				 data_format=df,
				 activation=tf.nn.relu,
				 use_bias=True,
				 kernel_initializer=tf.variance_scaling_initializer(),
				 bias_initializer=tf.zeros_initializer(),
				 kernel_regularizer=tf.contrib.layers.l2_regularizer(wd))
	net = max_pooling2d(inputs=net,
						pool_size=[3, 3],
						strides=2,
						data_format=df)
	net = fire_module(net, 16, 64, wd, df)
	net = fire_module(net, 16, 64, wd, df)
	net = fire_module(net, 32, 128, wd, df)
	net = max_pooling2d(inputs=net,
					 pool_size=[3, 3],
					 strides=2,
					 data_format=df)
	net = fire_module(net, 32, 128, wd, df)
	net = fire_module(net, 48, 192, wd, df)
	net = fire_module(net, 48, 192, wd, df)
	net = fire_module(net, 64, 256, wd, df)
	net = max_pooling2d(inputs=net,
					 pool_size=[3, 3],
					 strides=2,
					 data_format=df)
	net = fire_module(net, 64, 256, wd, df)
	net = dropout(inputs=net,
				  rate=0.5,
				  training=mode == tf.estimator.ModeKeys.TRAIN)
	net = conv2d(inputs=net,
				 filters=_NUM_CLASSES,
				 kernel_size=[1, 1],
				 strides=1, 
				 padding='valid', # no padding eqv. to pad=1 for 1x1 conv?
				 data_format=df,
				 activation=tf.nn.relu,
				 use_bias=True,
				 kernel_initializer=tf.initializers.random_normal(mean=0.0, stddev=0.01),
				 bias_initializer=tf.zeros_initializer(),
				 kernel_regularizer=tf.contrib.layers.l2_regularizer(wd))
	net = average_pooling2d(inputs=net,
							pool_size=[13, 13],
							strides=1,
							data_format=df)

	# TODO fix for data_format
	logits = tf.squeeze(net, [2,3])

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

	l2_loss = tf.losses.get_regularization_loss()
	tf.identity(l2_loss, name='l2_loss')
	tf.summary.scalar('l2_loss', l2_loss)

	loss = cross_entropy + l2_loss

	if mode == tf.estimator.ModeKeys.TRAIN:
		global_step = tf.train.get_or_create_global_step()
		learning_rate = tf.train.polynomial_decay(
			learning_rate=BASE_LR,
			global_step=global_step,
			decay_steps=MAX_STEPS,
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
		return tf.estimator.EstimatorSpec(
			mode=mode, loss=loss, train_op=train_op
		)

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

	return tf.estimator.EstimatorSpec(
		mode=mode, loss=loss, eval_metric_ops=metrics
	)

def get_filenames(is_training, data_dir):
	"""Return filenames for dataset."""
	if is_training:
		return [
			os.path.join(data_dir, 'train-%05d-of-01024' % i)
			for i in range(_NUM_TRAIN_FILES)]
	else:
		return [
			os.path.join(data_dir, 'validation-%05d-of-00128' % i)
			for i in range(128)]

def parse_serialized_example(serialized_example):
	# Dense features in Example proto.
	feature_map = {
		'image/encoded': tf.FixedLenFeature([], dtype=tf.string,
											default_value=''),
		'image/class/label': tf.FixedLenFeature([], dtype=tf.int64,
												default_value=-1),
		'image/class/text': tf.FixedLenFeature([], dtype=tf.string,
											   default_value=''),
	}
	sparse_float32 = tf.VarLenFeature(dtype=tf.float32)
	# Sparse features in Example proto.
	feature_map.update(
	{k: sparse_float32 for k in ['image/object/bbox/xmin',
								 'image/object/bbox/ymin',
								 'image/object/bbox/xmax',
								 'image/object/bbox/ymax']})

	features = tf.parse_single_example(serialized_example, feature_map)
	label = tf.cast(features['image/class/label'], dtype=tf.int32)
	label = label - 1

	xmin = tf.expand_dims(features['image/object/bbox/xmin'].values, 0)
	ymin = tf.expand_dims(features['image/object/bbox/ymin'].values, 0)
	xmax = tf.expand_dims(features['image/object/bbox/xmax'].values, 0)
	ymax = tf.expand_dims(features['image/object/bbox/ymax'].values, 0)

	# Note that we impose an ordering of (y, x) just to make life difficult.
	bbox = tf.concat([ymin, xmin, ymax, xmax], 0)

	# Force the variable number of bounding boxes into the shape
	# [1, num_boxes, coords].
	bbox = tf.expand_dims(bbox, 0)
	bbox = tf.transpose(bbox, [0, 2, 1])

	return features['image/encoded'], label, bbox

def preprocess_image(raw_image):
	image = tf.image.decode_jpeg(raw_image, channels=_NUM_CHANNELS)
	image = tf.image.resize_images(
		image, [_DEFAULT_IMAGE_SIZE, _DEFAULT_IMAGE_SIZE]
	)
#	image = tf.image.convert_image_dtype(image, tf.float32)
	image = tf.cast(image, dtype=tf.float32)
	image = tf.transpose(image, [2, 0, 1])
	return image

def parse_record(raw_record):
	raw_image, label, _ = parse_serialized_example(raw_record)
	image = preprocess_image(raw_image)
	return image, label

def input_fn(is_training, data_dir, batch_size, num_epochs=1):
	filenames = get_filenames(is_training, data_dir)
	dataset = tf.data.Dataset.from_tensor_slices(filenames)

	if is_training:
		# shuffle input files
		dataset = dataset.shuffle(buffer_size=_NUM_TRAIN_FILES)

	# convert to individual records
	# 10 files read and deserialized in parallel
	dataset = dataset.apply(
		tf.contrib.data.parallel_interleave(
			tf.data.TFRecordDataset, cycle_length=10
		)
	)

	# prefetch a batch at a time
	dataset = dataset.prefetch(buffer_size=batch_size)

	if is_training:
		# shuffle before repeat respects epoch boundaries
		dataset = dataset.shuffle(buffer_size=_SHUFFLE_BUFFER)

	dataset = dataset.repeat(num_epochs)

	if is_training:
		total_examples = num_epochs * _NUM_TRAIN_IMAGES
		# Force the number of batches to be divisible by the number of devices.
		# This prevents some devices from receiving batches while others do not,
		# which can lead to a lockup. This case will soon be handled directly by
		# distribution strategies, at which point this .take() operation will no
		# longer be needed.
		total_batches = total_examples // batch_size // NUM_GPUS * NUM_GPUS 
		dataset.take(total_batches * batch_size)

	# Parses the raw records into images and labels.
	dataset = dataset.apply(
		tf.contrib.data.map_and_batch(
			lambda value: parse_record(value),
			batch_size=batch_size,
			num_parallel_batches=1,
			drop_remainder=False))

	# ops between final prefetch and get_next call to iterator are sync.
	# prefetch again to background preprocessing work.
#	dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
	dataset = dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)

	return dataset

def train_input_fn(num_epochs):
	return input_fn(
		is_training=True,
		data_dir=DATA_DIR,
		batch_size=L_BATCH_SIZE,
		num_epochs=num_epochs
	)

def eval_input_fn():
	return input_fn(
		is_training=False,
		data_dir=DATA_DIR,
		batch_size=L_BATCH_SIZE,
		num_epochs=1
	)

tf.logging.set_verbosity( tf.logging.INFO )

session_config = tf.ConfigProto(allow_soft_placement=True)

config = tf.estimator.RunConfig(
	session_config=session_config,
	train_distribute=tf.contrib.distribute.MirroredStrategy(num_gpus=NUM_GPUS)
)

classifier = tf.estimator.Estimator(model_fn=model_fn, model_dir=MODEL_DIR,
	config=config, params={
		'weight_decay': WEIGHT_DECAY,
		'data_format': DATA_FORMAT
	})

#hooks = [tf_debug.LocalCLIDebugHook()]

for epoch in range(NUM_EPOCHS):
#	classifier.train(input_fn=lambda: train_input_fn(1), hooks=hooks)
	tf.logging.info("train epoch {}".format(epoch))
	classifier.train(input_fn=lambda: train_input_fn(1))
	tf.logging.info("eval epoch {}".format(epoch))
	classifier.evaluate(input_fn=eval_input_fn)
