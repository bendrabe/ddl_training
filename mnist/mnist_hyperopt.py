import numpy as np
import os
import random
import tensorflow as tf

winit_space = [
	("uniform_fanin", tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=True)),
	("uniform_fanavg", tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG', uniform=True)),
	("norm_fanin", tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False)),
	("norm_fanavg", tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG', uniform=False))
]
hu_space_min = 18
hu_space_max = 1024
act_space = [("sigmoid", tf.nn.sigmoid), ("tanh", tf.nn.tanh)]
batch_space = [20,100]
lr_space_min = 1.0e-3
lr_space_max = 10.0
t0_space_min = 300
t0_space_max = 30000
l2_space_min = 3.1e-7
l2_space_max = 3.1e-5

S = 256
min_epochs = 100
max_epochs = 1000
out_dir = "summary/mnist"

def nn_model_fn(features, labels, mode, params):
	input_layer = tf.reshape( features["x"], [-1, 28*28] )
	dense = tf.layers.dense(inputs=input_layer, units=params['hu'], activation=params['act'], kernel_initializer=params['winit'], kernel_regularizer=tf.contrib.layers.l2_regularizer(params['l2']))
	logits = tf.layers.dense(inputs=dense, units=10, kernel_initializer=tf.zeros_initializer(), kernel_regularizer=tf.contrib.layers.l2_regularizer(params['l2']))
	l2_loss = tf.losses.get_regularization_loss()

	predictions = {
		# Generate predictions (for PREDICT and EVAL mode)
		"classes": tf.argmax(input=logits, axis=1),
		# Add `softmax_tensor` to the graph. It is used for PREDICT and by the `logging_hook`.
		"probabilities": tf.nn.softmax(logits, name="softmax_tensor")
	}

	if mode == tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

	# Calculate Loss (for both TRAIN and EVAL modes)
	loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits) + l2_loss

	# Configure the Training Op (for TRAIN mode)
	if mode == tf.estimator.ModeKeys.TRAIN:
		#optimizer = tf.train.GradientDescentOptimizer(learning_rate=params['lr_tensor'])
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

tf.logging.set_verbosity(tf.logging.ERROR)

((data1, labels1), (data2, labels2)) = tf.keras.datasets.mnist.load_data()
data1 = data1/np.float32(255)
labels1 = labels1.astype(np.int32)  # not required
data2 = data2/np.float32(255)
labels2 = labels2.astype(np.int32)  # not required

train_data = data1[:10000]
train_labels = labels1[:10000]

valid_data = data2[:2000]
valid_labels = labels2[:2000]

test_data = data1[10000:]
test_labels = labels1[10000:]

for s in range(S):
	winit = random.choice(winit_space)

	log_hu = np.log(hu_space_min) + (np.log(hu_space_max) - np.log(hu_space_min))*np.random.rand()
	hu_float = np.exp(log_hu)
	hu = int(np.rint(hu_float))

	act = random.choice(act_space)

	batch_size = random.choice(batch_space)

	log_lr = np.log(lr_space_min) + (np.log(lr_space_max) - np.log(lr_space_min))*np.random.rand()
	lr = np.exp(log_lr)

	log_t0 = np.log(t0_space_min) + (np.log(t0_space_max) - np.log(t0_space_min))*np.random.rand()
	t0_float = np.exp(log_t0)
	t0 = int(np.rint(t0_float))

	log_l2 = np.log(l2_space_min) + (np.log(l2_space_max) - np.log(l2_space_min))*np.random.rand()
	l2 = np.exp(log_l2)

	hyperparams = {'winit': winit[1], 'hu': hu, 'act': act[1], 'bs': batch_size, 'lr': lr, 't0': t0, 'l2': l2}
	hyperparams_str = {'winit': winit[0], 'hu': hu, 'act': act[0], 'bs': batch_size, 'lr': lr, 't0': t0, 'l2': l2}

	mnist_classifier = tf.estimator.Estimator(model_fn=nn_model_fn, model_dir=out_dir+str(s), params=hyperparams)

	train_input_fn = tf.estimator.inputs.numpy_input_fn(
		x={"x": train_data},
		y=train_labels,
		batch_size=batch_size,
		num_epochs=1,
		shuffle=True)

	eval_input_fn = tf.estimator.inputs.numpy_input_fn(
		x={"x": valid_data},
		y=valid_labels,
		num_epochs=1,
		shuffle=False)

	max_val = 0.0
	max_epoch = 0
	epoch = 0
	done = False

	while not done:
		mnist_classifier.train( input_fn=train_input_fn )

		eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
		val = eval_results['accuracy']

		if val > max_val:
			max_val = val
			max_epoch = epoch

		if (epoch > max_epochs) or (epoch > min_epochs and val < max_val and max_epoch < 3*epoch/4):
			done = True

		epoch += 1

	with open(out_dir+"results.txt", "a") as f:
		f.write(str(hyperparams_str) + str(max_val) + "\n")
