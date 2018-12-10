#!/usr/bin/env python3
import os.path
import tensorflow as tf
import math
import numpy as np
from tqdm import tqdm
import time
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
from args import FLAGS

MODEL_DIR = 'models'
MODEL_NAME = 'fcn-vgg16'
MODEL_EXT = '.ckpt'
IMAGE_SHAPE = (160, 576)  # KITTI dataset uses 160x576 images
CLASSES_N = 2

if not os.path.isdir(MODEL_DIR):
    os.makedirs(MODEL_DIR)
    
# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)

print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
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

    # 1x1 convolutions to the encoder layers
    layer3_1x1 = conv_1x1(vgg_layer3_out, num_classes, 'layer3_1x1')
    layer4_1x1 = conv_1x1(vgg_layer4_out, num_classes, 'layer4_1x1')
    layer7_1x1 = conv_1x1(vgg_layer7_out, num_classes, 'layer7_1x1')

    # Upsample to decode into final image size
    layer7_up = up_sample(layer7_1x1, num_classes, 'layer7_up')

    layer4_skip = tf.add(layer7_up, layer4_1x1, name="layer4_skip")
    layer4_up = up_sample(layer4_skip, num_classes, 'layer4_up')

    layer3_skip = tf.add(layer4_up, layer3_1x1, name='layer3_skip')
    layer3_up = up_sample(layer3_skip, num_classes, 'layer3_up', kernel_size=(16, 16), strides=(8, 8))

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
    
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    cross_entropy_loss = tf.reduce_mean(cross_entropy)

    # Keeps track of the training steps
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(cross_entropy_loss)

    return logits, train_op, cross_entropy_loss

def save_model(saver, sess, epoch = None):
    file_name = MODEL_NAME
    if epoch is not None:
        file_name += '_ep_' + epoch
    file_name += MODEL_EXT
    file_path = os.path.join(MODEL_DIR, file_name)
    save_path = saver.save(sess, file_path)
    print("Model saved in path: {}".format(save_path))


def train_nn(sess, epochs, batch_size, get_batches_fn, batches_n, train_op, cross_entropy_loss, image_input,
             labels, keep_prob, learning_rate, l2_beta, save_model = False):
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
    :param l2_beta: TF Placeholder for l2 regularization beta
    """

    print('Training = (Epochs: {}, Batch Size: {}, Learning Rate: {}, Dropout: {}, L2 Beta: {})'.format(
        FLAGS.epochs,
        FLAGS.batch_size,
        FLAGS.learning_rate,
        FLAGS.dropout,
        FLAGS.l2_beta
    ))

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    if save_model:
        saver = tf.train.Saver()

    loss_log = []

    start = time.time()

    for epoch in range(epochs):

        curr_epoch = epoch + 1
        batches = tqdm(get_batches_fn(batch_size),
                       desc='Epoch {}/{} (Loss: N/A)'.format(curr_epoch, epochs),
                       unit='batches',
                       total=batches_n)

        losses = []

        for batch_images, batch_labels in batches:

            feed_dict = {
                image_input: batch_images,
                labels: batch_labels,
                keep_prob: (1 - FLAGS.dropout),
                learning_rate: FLAGS.learning_rate,
                l2_beta: FLAGS.l2_beta
            }

            _, loss = sess.run([train_op, cross_entropy_loss], feed_dict=feed_dict)

            losses.append(loss)
            training_loss = np.mean(losses)
            loss_log.append(training_loss)

            batches.set_description('Epoch {}/{} (Loss: {})'.format(curr_epoch, epochs, training_loss))
        
        if epoch % 5 == 0 and saver is not None:
            save_model(sess, saver, curr_epoch)

    elapsed = time.time() - start
    print("Training finished ({:.1f} s)".format(elapsed))

    if saver is not None:
        save_model(sess, saver)
    
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
    get_batches_fn, samples_n = helper.gen_batch_function(os.path.join(FLAGS.data_dir, 'data_road/training'), IMAGE_SHAPE)
    
    batches_n = int(math.ceil(float(samples_n) / FLAGS.batch_size))

    with tf.Session() as sess:

        labels = tf.placeholder(tf.float32, [None, None, None, CLASSES_N], 'input_labels')
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        l2_beta = tf.placeholder(tf.float32, name='l2_beta')

        image_input, keep_prob, layer3, layer4, layer7 = load_vgg(sess, vgg_path)
        model_output = layers(layer3, layer4, layer7, CLASSES_N)
        logits, train_op, cross_entropy_loss = optimize(model_output, labels, learning_rate, CLASSES_N)

        train_nn(sess, FLAGS.epochs, FLAGS.batch_size, get_batches_fn, batches_n, train_op, cross_entropy_loss, image_input,
                 labels, keep_prob, learning_rate, l2_beta, True)

        helper.save_inference_samples(FLAGS.runs_dir, FLAGS.data_dir, sess, IMAGE_SHAPE, logits, keep_prob, image_input)

def main(_):
    run_tests()
    run()

if __name__ == '__main__':
    tf.app.run()
