#!/usr/bin/env python3
import os.path
import tensorflow as tf
import math
import numpy as np
from tqdm import tqdm
import time
import helper
from distutils.version import LooseVersion
import project_tests as tests
from args import FLAGS
from datetime import datetime

LOGS_DIR = 'logs'
MODEL_DIR = 'models'
MODEL_NAME = 'fcn-vgg16'
MODEL_EXT = '.ckpt'
IMAGE_SHAPE = (160, 576)  # KITTI dataset uses 160x576 images
CLASSES_N = 2

if not os.path.isdir(MODEL_DIR):
    os.makedirs(MODEL_DIR)

if not os.path.isdir(LOGS_DIR):
    os.makedirs(LOGS_DIR)

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)

print('TensorFlow Version: {}'.format(tf.__version__))


def warn_msg(message):
    print("[Warning]: {}".format(message))


# Check for a GPU
if not tf.test.gpu_device_name():
    warn_msg('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


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


def conv_1x1(x, filters, name, inizializer=None, regularizer=None):
    return tf.layers.conv2d(x,
                            filters=filters,
                            kernel_size=(1, 1),
                            strides=(1, 1),
                            padding='same',
                            kernel_initializer=inizializer,
                            kernel_regularizer=regularizer,
                            name=name)


def up_sample(x, filters, name, kernel_size=(4, 4), strides=(2, 2), inizializer=None, regularizer=None):
    return tf.layers.conv2d_transpose(x,
                                      filters=filters,
                                      kernel_size=kernel_size,
                                      strides=strides,
                                      padding='same',
                                      kernel_initializer=inizializer,
                                      kernel_regularizer=regularizer,
                                      name=name)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    inizializer = tf.truncated_normal_initializer(stddev=FLAGS.w_std)
    regularizer = tf.contrib.layers.l2_regularizer(FLAGS.l2_reg)

    #vgg_layer3_scaled = tf.multiply(vgg_layer3_out, 0.0001, name='layer3_scaled')
    #vgg_layer4_scaled = tf.multiply(vgg_layer4_out, 0.01, name='layer4_scaled')

    # 1x1 convolutions to the encoder layers
    layer3_1x1 = conv_1x1(vgg_layer3_out, num_classes, 'layer3_1x1',
                          inizializer=inizializer, regularizer=regularizer)
    layer4_1x1 = conv_1x1(vgg_layer4_out, num_classes, 'layer4_1x1',
                          inizializer=inizializer, regularizer=regularizer)
    layer7_1x1 = conv_1x1(vgg_layer7_out, num_classes, 'layer7_1x1',
                          inizializer=inizializer, regularizer=regularizer)

    # Upsample to decode into final image size
    layer7_up = up_sample(layer7_1x1, num_classes, 'layer7_up',
                          inizializer=inizializer, regularizer=regularizer)

    layer4_skip = tf.add(layer7_up, layer4_1x1, name="layer4_skip")
    layer4_up = up_sample(layer4_skip, num_classes, 'layer4_up',
                          inizializer=inizializer, regularizer=regularizer)

    layer3_skip = tf.add(layer4_up, layer3_1x1, name='layer3_skip')
    layer3_up = up_sample(layer3_skip, num_classes, 'layer3_up', kernel_size=(
        16, 16), strides=(8, 8), inizializer=inizializer, regularizer=regularizer)

    return layer3_up


def optimize(nn_last_layer, labels, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param labels: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    logits = tf.reshape(nn_last_layer, (-1, num_classes))

    # sparse_softmax_cross_entropy_with_logits
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=labels)

    # Applies L2 regularization
    cross_entropy_loss = tf.reduce_mean(cross_entropy) + tf.losses.get_regularization_loss()

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(cross_entropy_loss)

    return logits, train_op, cross_entropy_loss


def write_model(saver, sess, epoch=None):
    file_name = MODEL_NAME
    if epoch is not None:
        file_name += '_ep_' + str(epoch)
    file_name += MODEL_EXT
    file_path = os.path.join(MODEL_DIR, file_name)
    save_path = saver.save(sess, file_path)
    print('Model saved in path: {}'.format(save_path))


def load_model(sess):
    saver = tf.train.Saver()
    file_name = MODEL_NAME + MODEL_EXT
    file_path = os.path.join(MODEL_DIR, file_name)
    if not os.path.isfile(file_path):
        raise ValueError('The model {} does not exist'.format(file_path))
    saver.restore(sess, file_name)
    print('Model restored from path: {}'.format(file_path))


def train_nn(sess, epochs, batch_size, get_batches_fn, batches_n, train_op, cross_entropy_loss, image_input,
             labels, keep_prob, learning_rate, save_model=False, tensorboard=False):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param image_input: TF Placeholder for input images
    :param labels: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """

    print('Training = (Epochs: {}, Batch Size: {}, Learning Rate: {}, Dropout: {}, L2 Reg: {})'.format(
        FLAGS.epochs,
        FLAGS.batch_size,
        FLAGS.learning_rate,
        FLAGS.dropout,
        FLAGS.l2_reg
    ))

    if save_model and FLAGS.restore:
        load_model(sess)
    else:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

    if save_model:
        saver = tf.train.Saver()
    else:
        saver = None

    if tensorboard:
        tf.summary.scalar('learning_rate', learning_rate)
        tf.summary.scalar('total_loss', cross_entropy_loss) 

        summary = tf.summary.merge_all()
        writer_folder = os.path.join(LOGS_DIR, datetime.now().strftime('%Y%m%d_%H%M%S'))
        writer = tf.summary.FileWriter(writer_folder, sess.graph)
    else:
        summary = None

    loss_log = []

    start = time.time()

    step = 0

    for epoch in range(epochs):

        curr_epoch = epoch + 1
        batches = tqdm(get_batches_fn(batch_size),
                       desc='Epoch {}/{} (Loss: N/A, Step: N/A)'.format(curr_epoch, epochs),
                       unit='batch',
                       total=batches_n)

        losses = []

        for batch_images, batch_labels in batches:

            feed_dict = {
                image_input: batch_images,
                labels: batch_labels,
                keep_prob: (1 - FLAGS.dropout),
                learning_rate: FLAGS.learning_rate
            }

            step += len(batch_images)

            if summary is None:
                _, loss = sess.run([train_op, cross_entropy_loss], feed_dict=feed_dict)
            else:
                _, loss, train_summary = sess.run([train_op, cross_entropy_loss, summary], feed_dict=feed_dict)
                writer.add_summary(train_summary, global_step=step)

            losses.append(loss)
            training_loss = np.mean(losses)
            loss_log.append(training_loss)

            batches.set_description('Epoch {}/{} (Loss: {:.4f}, Step: {})'.format(curr_epoch, epochs, training_loss, step))

        if epoch % 5 == 0 and saver is not None:
            write_model(sess, saver, curr_epoch)

    elapsed = time.time() - start
    print("Training finished ({:.1f} s)".format(elapsed))

    if saver is not None:
        write_model(sess, saver)

    return loss_log


def run_tests():
    helper.maybe_download_pretrained_vgg(FLAGS.data_dir)
    tests.test_for_kitti_dataset(FLAGS.data_dir)
    tests.test_load_vgg(load_vgg, tf)
    tests.test_layers(layers)
    tests.test_optimize(optimize)
    tests.test_train_nn(train_nn)


def run():

    # Download pretrained vgg model
    vgg_path = helper.maybe_download_pretrained_vgg(FLAGS.data_dir)
    # Create function to get batches
    get_batches_fn, samples_n = helper.gen_batch_function(
        os.path.join(FLAGS.data_dir, 'data_road/training'), IMAGE_SHAPE)

    batches_n = int(math.ceil(float(samples_n) / FLAGS.batch_size))

    config = None

    if FLAGS.cpu:
        warn_msg("Forcing CPU usage")
        config = tf.ConfigProto(device_count={'GPU': 0})

    with tf.Session(config=config) as sess:

        labels = tf.placeholder(
            tf.float32, [None, None, None, CLASSES_N], 'input_labels')
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')

        image_input, keep_prob, layer3, layer4, layer7 = load_vgg(sess, vgg_path)
        model_output = layers(layer3, layer4, layer7, CLASSES_N)

        logits, train_op, cross_entropy_loss = optimize(model_output,
                                                                     labels,
                                                                     learning_rate,
                                                                     CLASSES_N)

        train_nn(sess,
                 FLAGS.epochs,
                 FLAGS.batch_size,
                 get_batches_fn, batches_n,
                 train_op,
                 cross_entropy_loss,
                 image_input,
                 labels,
                 keep_prob,
                 learning_rate,
                 True,
                 True)

        helper.save_inference_samples(
            FLAGS.runs_dir, FLAGS.data_dir, sess, IMAGE_SHAPE, logits, keep_prob, image_input)


def main(_):
    run_tests()
    run()


if __name__ == '__main__':
    tf.app.run()
