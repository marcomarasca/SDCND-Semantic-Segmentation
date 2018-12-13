#!/usr/bin/env python3
from args import FLAGS

import os.path
import tensorflow as tf
import math
import scipy.misc
import time
import imageio
import numpy as np
from tqdm import tqdm
from tqdm import trange
from distutils.version import LooseVersion

import helper
import project_tests as tests
import augmentation

KERNEL_STDEV = 0.01
SCALE_L_3 = 0.0001
SCALE_L_4 = 0.01
MODELS_LIMIT = 5
MODELS_FREQ = 5
TENSORBOARD_FREQ = 5
TENSORBOARD_MAX_IMG = 3

IMAGE_SHAPE = (160, 576)
CLASSES_N = 2

# Check TensorFlow Version
assert LooseVersion(
    tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(
        tf.__version__)

print('TensorFlow Version: {}'.format(tf.__version__))


def warn_msg(message):
    print("[Warning]: {}".format(message))


# Check for a GPU
if not tf.test.gpu_device_name():
    warn_msg('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def _conv_1x1(x, filters, name, regularizer=None):
    """
    Applies a 1x1 convolution to the given input
    :param x: The input
    :param filters: Number of filters
    :param name: The name of the tensor
    :param regularizer: Optional regularizer for the kernel
    :return: Tensor with the 1x1 convolution
    """
    return tf.layers.conv2d(
        x,
        filters=filters,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding='same',
        kernel_initializer=tf.truncated_normal_initializer(stddev=KERNEL_STDEV, seed=FLAGS.seed),
        kernel_regularizer=regularizer,
        name=name)


def _up_sample(x, filters, name, kernel_size, strides, regularizer=None):
    """
    Up sample the given input using a conv2d_transpose convolution
    :param x: The input
    :param filters: Number of filters
    :param name: The name of the tensor
    :param kernel_size: The kernel size dimensions
    :param stride: The stride to apply
    :param regularizer: Optional regularizer for the kernel
    :return: Tensor with the upsampled input
    """
    return tf.layers.conv2d_transpose(
        x,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding='same',
        kernel_initializer=tf.truncated_normal_initializer(stddev=KERNEL_STDEV, seed=FLAGS.seed),
        kernel_regularizer=regularizer,
        name=name)


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """

    tf.saved_model.loader.load(sess, ['vgg16'], vgg_path)
    graph = tf.get_default_graph()

    image_input = graph.get_tensor_by_name('image_input:0')
    keep_prob = graph.get_tensor_by_name('keep_prob:0')
    layer3_out = graph.get_tensor_by_name('layer3_out:0')
    layer4_out = graph.get_tensor_by_name('layer4_out:0')
    layer7_out = graph.get_tensor_by_name('layer7_out:0')

    return image_input, keep_prob, layer3_out, layer4_out, layer7_out


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """

    l2_reg = tf.contrib.layers.l2_regularizer(FLAGS.l2_reg)

    if FLAGS.scale:
        # Scale layers (See optimized at-once architecture from the original implementation
        # of FCN-8s PASCAL at-once: https://github.com/shelhamer/fcn.berkeleyvision.org)
        vgg_layer3_out = tf.multiply(vgg_layer3_out, SCALE_L_3, name='layer3_scaled')
        vgg_layer4_out = tf.multiply(vgg_layer4_out, SCALE_L_4, name='layer4_scaled')

    # 1x1 convolutions to the encoder layers
    layer3_1x1 = _conv_1x1(vgg_layer3_out, num_classes, 'layer3_1x1', regularizer=l2_reg)
    layer4_1x1 = _conv_1x1(vgg_layer4_out, num_classes, 'layer4_1x1', regularizer=l2_reg)
    layer7_1x1 = _conv_1x1(vgg_layer7_out, num_classes, 'layer7_1x1', regularizer=l2_reg)

    # Upsample to decode into final image size
    layer7_up = _up_sample(layer7_1x1, num_classes, 'layer7_up', (4, 4), (2, 2), regularizer=l2_reg)

    # Skip layer
    layer4_skip = tf.add(layer7_up, layer4_1x1, name="layer4_skip")
    layer4_up = _up_sample(layer4_skip, num_classes, 'layer4_up', (4, 4), (2, 2), regularizer=l2_reg)

    # Skip layer
    layer3_skip = tf.add(layer4_up, layer3_1x1, name='layer3_skip')
    layer3_up = _up_sample(layer3_skip, num_classes, 'layer3_up', (16, 16), (8, 8), regularizer=l2_reg)

    return layer3_up


def optimize(nn_last_layer, labels, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param labels: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss, global_step)
    """
    logits = tf.reshape(nn_last_layer, (-1, num_classes))

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)

    # Applies L2 regularization
    cross_entropy_loss = tf.reduce_mean(cross_entropy) + tf.losses.get_regularization_loss()

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=FLAGS.eps)

    global_step = tf.Variable(initial_value=0, trainable=False, name='global_step')

    train_op = optimizer.minimize(cross_entropy_loss, global_step=global_step)

    return logits, train_op, cross_entropy_loss, global_step


def metrics(output_softmax, labels, num_classes):
    """
    Builds the metrics for the model, including IoU and accuracy

    :param output_softmax: TF Tensor containing the sofmax operation on the last layer in the neural network 
                           before the decoder
    :param labels: TF Placeholder for the correct label image
    :param num_classes: Number of classes to classify
    :return: A tuple (metrics and metrics_reset_op). The metrics is a dictionary with metrics (iou and acc) 
             that each contain a tuple with the tensor value and the update tensor operation. The metric_reset_op
             is an operation that reinizializes the internal counters of the metrics so that they can be reset at
             the beginning of an epoch
    """
    logits_argmax = tf.argmax(output_softmax, axis=-1, name='output_argmax')
    labels_argmax = tf.argmax(labels, axis=-1, name='labels_argmax')

    metrics = {}

    with tf.variable_scope('metrics') as scope:
        metrics['iou'] = (tf.metrics.mean_iou(labels_argmax, logits_argmax, num_classes))
        metrics['acc'] = (tf.metrics.accuracy(labels_argmax, logits_argmax))

    # Creates a reset operation for the metrics to be run at the beginning of each epoch
    # See https://steemit.com/machine-learning/@ronny.rest/avoiding-headaches-with-tf-metrics
    metrics_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metrics")
    metrics_reset_op = tf.variables_initializer(var_list=metrics_vars)

    return metrics, metrics_reset_op


def prediction(model_output):
    """
    Builds the prediction tensors for the model

    :param model_output: TF Tensor of the last layer in the neural network before the decoder
    :return: A tuple (output_softmax, prediction_op)
    """
    output_softmax = tf.nn.softmax(model_output, name='output_softmax')
    prediction_class = tf.cast(tf.greater(output_softmax, 0.5), dtype=tf.float32)

    return output_softmax, tf.cast(tf.argmax(prediction_class, axis=3), dtype=tf.uint8)


def train_nn(sess,
             global_step,
             epochs,
             batch_size,
             get_batches_fn,
             batches_n,
             train_op,
             cross_entropy_loss,
             prediction_op,
             metrics,
             metrics_reset_op,
             image_input,
             labels,
             keep_prob,
             learning_rate,
             save_model_freq=None,
             tensorboard_freq=None):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param global_step: TF Placeholder containing the global step
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param batches_n: Number of batches to cover all the samples
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param prediction_op: TF Tensor for the prediction class (index)
    :param metrics: Dictionary with the evaluation metrics
    :param metric_reset_op: TF Tensor used to reset the metrics counters
    :param image_input: TF Placeholder for input images
    :param labels: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    :param save_model_freq: The frequency to save the model to disk, None to disable
    :param tensorboard_freq: The frequency to push the summaries to tensorboard, None to disable
    """

    model_folder = helper.model_folder()

    if save_model_freq and helper.checkpoint_exists(model_folder):
        print('Checkpoint exists, restoring model from {}'.format(model_folder))
        helper.load_model(sess, model_folder)
    else:
        sess.run(tf.global_variables_initializer())

    sess.run(tf.local_variables_initializer())

    if save_model_freq:
        saver = tf.train.Saver(max_to_keep=MODELS_LIMIT)

    iou_mean, iou_op = metrics['iou']
    acc_mean, acc_op = metrics['acc']

    # Evaluate current step
    step = global_step.eval(session=sess)
    start_step = step

    if tensorboard_freq:
        # Creates the tensorboard writer
        train_writer = helper.summary_writer(sess, model_folder)

        # Gets the batch of images/labels to feed to the image summary op
        summary_images, summary_labels = helper.image_summary_batch(
            os.path.join(FLAGS.data_dir, 'data_road', 'training'), IMAGE_SHAPE, TENSORBOARD_MAX_IMG)

        # Setup the summary ops
        summary_op, image_summary_op = helper.setup_summaries(sess, train_writer, image_input, labels, keep_prob,
                                                              cross_entropy_loss, prediction_op, iou_mean, acc_mean,
                                                              summary_images, summary_labels, step, CLASSES_N)

    training_log = []

    print('Model folder: {}'.format(model_folder))
    print(
        'Training (First batch: {}, Epochs: {}, Batch Size: {}, Learning Rate: {}, Dropout: {}, L2 Reg: {}, Eps: {}, Scaling: {})'
        .format(step + 1, FLAGS.epochs, FLAGS.batch_size, FLAGS.learning_rate, FLAGS.dropout, FLAGS.l2_reg, FLAGS.eps,
                'ON' if FLAGS.scale else 'OFF'))

    best_loss = 9999
    ep_loss_incr = 0

    start = time.time()

    for epoch in range(epochs):

        total_loss = 0
        mean_loss = 9999
        mean_acc = 0
        mean_iou = 0
        images_n = 0

        # Resets the metrics variables at the beginning of the epoch
        sess.run(metrics_reset_op)

        batches = tqdm(
            get_batches_fn(batch_size),
            desc='Epoch {}/{} (Step: {}, Samples: N/A, Loss: N/A, Acc: N/A, IoU: N/A)'.format(epoch + 1, epochs, step),
            unit='batches',
            total=batches_n)

        for batch_images, batch_labels in batches:

            feed_dict = {
                image_input: batch_images,
                labels: batch_labels,
                keep_prob: (1.0 - FLAGS.dropout),
                learning_rate: FLAGS.learning_rate
            }

            # Train
            _ = sess.run(train_op, feed_dict=feed_dict)

            images_n += len(batch_images)

            # Evaluate
            loss, _, mean_iou, _, mean_acc = sess.run([cross_entropy_loss, iou_op, iou_mean, acc_op, acc_mean],
                                                      feed_dict={
                                                          image_input: batch_images,
                                                          labels: batch_labels,
                                                          keep_prob: 1.0
                                                      })

            step = global_step.eval(session=sess)

            total_loss += loss * len(batch_images)
            mean_loss = total_loss / images_n

            # Saves metrics for tensorboard
            if tensorboard_freq:

                # Updates the summary according to frequency
                if step % tensorboard_freq == 0:
                    training_summary = sess.run(
                        summary_op, feed_dict={
                            image_input: batch_images,
                            labels: batch_labels,
                            keep_prob: 1.0
                        })
                    train_writer.add_summary(training_summary, global_step=step)

                # Writes the image every epoch
                if step % batches_n == 0:
                    image_pred_summary = sess.run(
                        image_summary_op,
                        feed_dict={
                            image_input: summary_images,
                            labels: summary_labels,
                            keep_prob: 1.0
                        })
                    train_writer.add_summary(image_pred_summary, global_step=step)
                    train_writer.flush()

            batches.set_description(
                'Epoch {}/{} (Step: {}, Samples: {}, Loss: {:.4f}, Acc: {:.4f}, IoU: {:.4f})'.format(
                    epoch + 1, epochs, step, images_n, mean_loss, mean_acc, mean_iou))

        training_log.append((mean_loss, mean_acc, mean_iou))

        if mean_loss < best_loss:
            ep_loss_incr = 0
            best_loss = mean_loss
        else:
            ep_loss_incr += 1

        if FLAGS.early_stopping is not None and ep_loss_incr >= FLAGS.early_stopping:
            print('Early Stopping Triggered (Loss not decreasing in the last {} epochs)'.format(ep_loss_incr))
            break

        if save_model_freq and (epoch + 1) % save_model_freq == 0:
            helper.save_model(sess, saver, model_folder, global_step)
            log_data = helper.to_log_data(training_log, start_step, step, batches_n)
            helper.save_log(log_data, model_folder)
            helper.plot_log(log_data, model_folder)

    elapsed = time.time() - start

    print('Training Completed ({:.1f} s): Last batch: {}, Loss: {:.4f}, Acc: {:.4f}, IoU: {:.4f}'.format(
        elapsed, step, mean_loss, mean_acc, mean_iou))

    if save_model_freq:
        helper.save_model(sess, saver, model_folder, global_step)
        log_data = helper.to_log_data(training_log, start_step, step, batches_n)
        helper.save_log(log_data, model_folder)
        helper.plot_log(log_data, model_folder)


def run_tests():
    helper.maybe_download_pretrained_vgg(FLAGS.data_dir)
    tests.test_for_kitti_dataset(FLAGS.data_dir)
    tests.test_load_vgg(load_vgg, tf)
    tests.test_layers(layers)
    tests.test_optimize(optimize)
    tests.test_train_nn(train_nn)


def get_config():
    config = None

    if FLAGS.cpu:
        warn_msg("Forcing CPU usage")
        config = tf.ConfigProto(device_count={'GPU': 0})

    return config


def process_image(file_path):

    if not os.path.isfile(file_path):
        raise ValueError('The file {} does not exist'.format(file_path))

    images_folder = os.path.join(FLAGS.runs_dir, 'images')

    if not os.path.isdir(images_folder):
        os.makedirs(images_folder)

    vgg_path = helper.maybe_download_pretrained_vgg(FLAGS.data_dir)

    with tf.Session(config=get_config()) as sess:
        image_input, keep_prob, layer3, layer4, layer7 = load_vgg(sess, vgg_path)
        model_output = layers(layer3, layer4, layer7, CLASSES_N)
        logits = tf.reshape(model_output, (-1, CLASSES_N))

        helper.load_model(sess, helper.model_folder())

        print('Processing image: {}'.format(file_path))
        name, image = helper.process_image_file(file_path, sess, logits, keep_prob, image_input, IMAGE_SHAPE)
        scipy.misc.imsave(os.path.join(images_folder, name), image)


def process_video(file_path):

    if not os.path.isfile(file_path):
        raise ValueError('The file {} does not exist'.format(file_path))

    videos_folder = os.path.join(FLAGS.runs_dir, 'videos')

    if not os.path.isdir(videos_folder):
        os.makedirs(videos_folder)

    video_output = os.path.join(videos_folder, os.path.basename(file_path))

    vgg_path = helper.maybe_download_pretrained_vgg(FLAGS.data_dir)

    with tf.Session(config=get_config()) as sess:
        image_input, keep_prob, layer3, layer4, layer7 = load_vgg(sess, vgg_path)
        model_output = layers(layer3, layer4, layer7, CLASSES_N)
        logits = tf.reshape(model_output, (-1, CLASSES_N))

        helper.load_model(sess, helper.model_folder())

        reader = imageio.get_reader(file_path)
        fps = reader.get_meta_data()['fps']
        writer = imageio.get_writer(video_output, fps=fps)

        for frame in tqdm(reader, desc='Processing Video', unit='frames'):
            frame_processed = helper.process_image(frame, sess, logits, keep_prob, image_input, IMAGE_SHAPE)
            writer.append_data(frame_processed)

        writer.close()


def run_testing():

    vgg_path = helper.maybe_download_pretrained_vgg(FLAGS.data_dir)

    with tf.Session(config=get_config()) as sess:
        image_input, keep_prob, layer3, layer4, layer7 = load_vgg(sess, vgg_path)
        model_output = layers(layer3, layer4, layer7, CLASSES_N)
        logits = tf.reshape(model_output, (-1, CLASSES_N))

        helper.load_model(sess, helper.model_folder())

        helper.save_inference_samples(FLAGS.runs_dir, FLAGS.data_dir, sess, IMAGE_SHAPE, logits, keep_prob, image_input)


def run():

    # Download pretrained vgg model
    vgg_path = helper.maybe_download_pretrained_vgg(FLAGS.data_dir)
    # Create function to get batches
    dataset_path = os.path.join(FLAGS.data_dir, 'data_road', 'training')
    get_batches_fn, samples_n = helper.gen_batch_function(dataset_path, IMAGE_SHAPE)

    batches_n = int(math.ceil(float(samples_n) / FLAGS.batch_size))

    with tf.Session(config=get_config()) as sess:

        labels = tf.placeholder(tf.float32, [None, None, None, CLASSES_N], 'input_labels')
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')

        image_input, keep_prob, layer3, layer4, layer7 = load_vgg(sess, vgg_path)
        model_output = layers(layer3, layer4, layer7, CLASSES_N)

        logits, train_op, cross_entropy_loss, global_step = optimize(model_output, labels, learning_rate, CLASSES_N)

        output_softmax, prediction_op = prediction(model_output)

        metrics_dict, metrics_reset_op = metrics(output_softmax, labels, CLASSES_N)

        train_nn(sess, global_step, FLAGS.epochs, FLAGS.batch_size, get_batches_fn, batches_n, train_op,
                 cross_entropy_loss, prediction_op, metrics_dict, metrics_reset_op, image_input, labels, keep_prob,
                 learning_rate, MODELS_FREQ, TENSORBOARD_FREQ)

        helper.save_inference_samples(FLAGS.runs_dir, FLAGS.data_dir, sess, IMAGE_SHAPE, logits, keep_prob, image_input)


def main(_):

    # Set a seed for reproducibility
    if FLAGS.seed is not None:
        np.random.seed(FLAGS.seed)
        tf.set_random_seed(FLAGS.seed)

    if FLAGS.augment:
        augmentation.augment_dataset(os.path.join(FLAGS.data_dir, 'data_road', 'training'), FLAGS.augment)
        return

    if FLAGS.tests:
        run_tests()
    if FLAGS.image:
        process_image(FLAGS.image)
    elif FLAGS.video:
        process_video(FLAGS.video)
    elif FLAGS.train:
        run()
    else:
        run_testing()


if __name__ == '__main__':
    tf.app.run()
