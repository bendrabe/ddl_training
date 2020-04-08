import os
import tensorflow as tf

_NUM_TRAIN_FILES=1024
_NUM_TRAIN_IMAGES=1281167
_SHUFFLE_BUFFER=10000
_DEFAULT_IMAGE_SIZE = 227
_NUM_CHANNELS = 3
_NUM_CLASSES = 1000
_RESIZE_MIN = 256
_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94
CHANNEL_MEANS = [_R_MEAN, _G_MEAN, _B_MEAN]
_R_STD = 58.39
_G_STD = 57.12
_B_STD = 57.38
CHANNEL_STDS = [_R_STD, _G_STD, _B_STD]

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

def _central_crop(image, crop_height, crop_width):
    """Performs central crops of the given image list.
    Args:
        image: a 3-D image tensor
        crop_height: the height of the image following the crop.
        crop_width: the width of the image following the crop.
    Returns:
        3-D tensor with cropped image.
    """
    shape = tf.shape(input=image)
    height, width = shape[0], shape[1]

    amount_to_be_cropped_h = (height - crop_height)
    crop_top = amount_to_be_cropped_h // 2
    amount_to_be_cropped_w = (width - crop_width)
    crop_left = amount_to_be_cropped_w // 2
    return tf.slice(
        image, [crop_top, crop_left, 0], [crop_height, crop_width, -1])

def _smallest_size_at_least(height, width, resize_min):
    """Computes new shape with the smallest side equal to `smallest_side`.

    Computes new shape with the smallest side equal to `smallest_side` while
    preserving the original aspect ratio.

    Args:
        height: an int32 scalar tensor indicating the current height.
        width: an int32 scalar tensor indicating the current width.
        resize_min: A python integer or scalar `Tensor` indicating the size of
            the smallest side after resize.

    Returns:
        new_height: an int32 scalar tensor indicating the new height.
        new_width: an int32 scalar tensor indicating the new width.
    """
    resize_min = tf.cast(resize_min, tf.float32)

    # Convert to floats to make subsequent calculations go smoothly.
    height, width = tf.cast(height, tf.float32), tf.cast(width, tf.float32)

    smaller_dim = tf.minimum(height, width)
    scale_ratio = resize_min / smaller_dim

    # Convert back to ints to make heights and widths that TF ops will accept.
    new_height = tf.cast(height * scale_ratio, tf.int32)
    new_width = tf.cast(width * scale_ratio, tf.int32)

    return new_height, new_width

def _aspect_preserving_resize(image, resize_min):
    """Resize images preserving the original aspect ratio.
    Args:
        image: A 3-D image `Tensor`.
        resize_min: A python integer or scalar `Tensor` indicating the size of
            the smallest side after resize.
    Returns:
        resized_image: A 3-D tensor containing the resized image.
    """
    shape = tf.shape(input=image)
    height, width = shape[0], shape[1]

    new_height, new_width = _smallest_size_at_least(height, width, resize_min)

    return _resize_image(image, new_height, new_width)

def _resize_image(image, height, width):
    """Simple wrapper around tf.resize_images.

    This is primarily to make sure we use the same `ResizeMethod` and other
    details each time.

    Args:
        image: A 3-D image `Tensor`.
        height: The target height for the resized image.
        width: The target width for the resized image.
    Returns:
        resized_image: A 3-D tensor containing the resized image. The first two
            dimensions have the shape [height, width].
    """
    return tf.compat.v1.image.resize(
    #return tf.image.resize_images(
        image, [height, width], method=tf.image.ResizeMethod.BILINEAR,
        align_corners=False)

def _central_crop(image, crop_height, crop_width):
    """Performs central crops of the given image list.
    Args:
        image: a 3-D image tensor
        crop_height: the height of the image following the crop.
        crop_width: the width of the image following the crop.
    Returns:
        3-D tensor with cropped image.
    """
    shape = tf.shape(input=image)
    height, width = shape[0], shape[1]

    amount_to_be_cropped_h = (height - crop_height)
    crop_top = amount_to_be_cropped_h // 2
    amount_to_be_cropped_w = (width - crop_width)
    crop_left = amount_to_be_cropped_w // 2
    return tf.slice(
        image, [crop_top, crop_left, 0], [crop_height, crop_width, -1])

def _squeeze_crop(image):
    image = _aspect_preserving_resize(image, _RESIZE_MIN)
    image = tf.image.random_crop(image, [_DEFAULT_IMAGE_SIZE, _DEFAULT_IMAGE_SIZE, _NUM_CHANNELS])
    #image = tf.random_crop(image, [_DEFAULT_IMAGE_SIZE, _DEFAULT_IMAGE_SIZE, _NUM_CHANNELS])
    return image

def _resnet_crop(image):
    bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
    shape = tf.shape(image)
    sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
        shape,
        bounding_boxes=bbox,
        min_object_covered=0,
        aspect_ratio_range=[0.75, 1.33],
        area_range=[0.08, 1.0],
        max_attempts=100,
        use_image_if_no_bounding_boxes=True)
    bbox_begin, bbox_size, _ = sample_distorted_bounding_box

    # Reassemble the bounding box in the format the crop op requires.
    offset_height, offset_width, _ = tf.unstack(bbox_begin)
    target_height, target_width, _ = tf.unstack(bbox_size)

    cropped = tf.image.crop_to_bounding_box(
        image,
        offset_height=offset_height,
        offset_width=offset_width,
        target_height=target_height,
        target_width=target_width)

    # Flip to add a little more random distortion in.
    cropped = tf.image.random_flip_left_right(cropped)
    return cropped

def preprocess_image(raw_image, is_training, crop, std):
    image = tf.image.decode_jpeg(raw_image, channels=_NUM_CHANNELS)

    if is_training:
        if crop == "squeeze":
            image = _squeeze_crop(image)
        else:
            image = _resnet_crop(image)
            image = _resize_image(image, _DEFAULT_IMAGE_SIZE, _DEFAULT_IMAGE_SIZE)
    else:
        image = _aspect_preserving_resize(image, _RESIZE_MIN)
        image = _central_crop(image, _DEFAULT_IMAGE_SIZE, _DEFAULT_IMAGE_SIZE)

    means = tf.broadcast_to(CHANNEL_MEANS, tf.shape(image))
    image = image - means

    if std:
        stds = tf.broadcast_to(CHANNEL_STDS, tf.shape(image))
        image = image / stds

    # convert from NHWC to NCHW and cast last
    image = tf.transpose(image, [2, 0, 1])
    image = tf.cast(image, dtype=tf.float32)
    return image

def parse_record(raw_record, is_training, crop, std):
    raw_image, label, _ = parse_serialized_example(raw_record)
    image = preprocess_image(raw_image, is_training, crop, std)
    label = tf.one_hot(label, depth=_NUM_CLASSES)
    return image, label

def get_real_input_fn(is_training, data_dir, batch_size, crop, std):
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
            lambda value: parse_record(value, is_training, crop, std),
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

def get_label(file_path):
    # convert the path to a list of path components
    parts = tf.strings.split([file_path], os.path.sep)
    # The second to last is the class-directory
    label = tf.strings.to_number(parts.values[-2], out_type=tf.int32)
    return label

def process_path(file_path, std):
    label = get_label(file_path)
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = preprocess_image(img, is_training=False, crop="squeeze", std=std)
    label = tf.one_hot(label, depth=_NUM_CLASSES)
    return img, label

def get_imagenet2_inputfn(data_dir, batch_size, std):
    dataset = tf.data.Dataset.list_files(os.path.join(data_dir, '*/*'))
    dataset = dataset.apply(
        tf.contrib.data.map_and_batch(
            lambda file_path: process_path(file_path, std),
            batch_size=batch_size,
            num_parallel_batches=1,
            drop_remainder=False))
    dataset = dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)
    return dataset
