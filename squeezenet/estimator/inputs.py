import os
import tensorflow as tf

_NUM_TRAIN_FILES=1024
_NUM_TRAIN_IMAGES=1281167
_SHUFFLE_BUFFER=10000
_DEFAULT_IMAGE_SIZE = 227
_NUM_CHANNELS = 3

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
#    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.cast(image, dtype=tf.float32)
    image = tf.transpose(image, [2, 0, 1])
    return image

def parse_record(raw_record):
    raw_image, label, _ = parse_serialized_example(raw_record)
    image = preprocess_image(raw_image)
    return image, label

def get_real_input_fn(is_training, data_dir, batch_size):
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
        # only repeat if training
        dataset = dataset.repeat()

    # Parses the raw records into images and labels.
    dataset = dataset.apply(
        tf.contrib.data.map_and_batch(
            lambda value: parse_record(value),
            batch_size=batch_size,
            num_parallel_batches=1,
            drop_remainder=False))

    # ops between final prefetch and get_next call to iterator are sync.
    # prefetch again to background preprocessing work.
    dataset = dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)

    return dataset

def get_synth_input_fn(shape, num_classes, dtype=tf.float32):
    """Returns an input function that returns a dataset with random data.
    This input_fn returns a data set that iterates over a set of random data and
    bypasses all preprocessing, e.g. jpeg decode and copy. The host to device
    copy is still included. This used to find the upper throughput bound when
    tunning the full input pipeline.
    Args:
        shape: Input image shape to generate (as list).
        num_classes: Number of classes that should be represented in the fake labels
            tensor
        dtype: Data type for features/images.
    Returns:
        An input_fn that can be used in place of a real one to return a dataset
        that can be used for iteration.
    """
    def input_fn(is_training, data_dir, batch_size, *args, **kwargs):
        """Returns dataset filled with random data."""
        # Synthetic input should be within [0, 255].
        inputs = tf.truncated_normal(
                [batch_size] + shape,
                dtype=dtype,
                mean=127,
                stddev=60,
                name='synthetic_inputs')
        inputs = tf.transpose(inputs, [0, 3, 1, 2])

        labels = tf.random_uniform(
                [batch_size],
                minval=0,
                maxval=num_classes - 1,
                dtype=tf.int32,
                name='synthetic_labels')

        steps_per_epoch = ((_NUM_TRAIN_IMAGES - 1 ) // batch_size) + 1
        data = tf.data.Dataset.from_tensors((inputs, labels)).repeat()
        data = data.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)
        return data

    return input_fn
