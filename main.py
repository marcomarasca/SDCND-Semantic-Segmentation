#!/usr/bin/env python3
import os.path
import tensorflow as tf
import math
import numpy as np
from tqdm import tqdm
from tqdm import trange
import time
import helper
from distutils.version import LooseVersion
import project_tests as tests
from args import FLAGS
from datetime import datetime
import pickle
import scipy.misc
import cv2
import matplotlib as mpl
# For plotting without a screen
mpl.use('Agg')
import matplotlib.pyplot as plt

LOGS_DIR = 'logs'
MODELS_LIMIT = 20
MODEL_DIR = 'models'
MODEL_NAME = 'fcn-vgg16'
MODEL_EXT = '.ckpt'
IMAGE_SHAPE = (160, 576)
CLASSES_N = 2

if not os.path.isdir(MODEL_DIR):
    os.makedirs(MODEL_DIR)

if not os.path.isdir(LOGS_DIR):
    os.makedirs(LOGS_DIR)

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


def _assert_folder_exists(folder):
    if not os.path.isdir(folder):
        raise ValueError('The folder {} does not exist'.format(folder))


def _save_model(sess, saver, model_folder, global_step):
    model_path = _model_checkpoint(model_folder)
    save_path = saver.save(sess, model_path, global_step=global_step)
    print('Model saved in path: {}'.format(save_path))


def _load_model(sess, model_folder):
    _assert_folder_exists(model_folder)
    saver = tf.train.Saver()
    # Restore the latest checkpoint
    model_path = tf.train.latest_checkpoint(model_folder)
    saver.restore(sess, model_path)
    print('Model restored from path: {}'.format(model_path))


def _to_log_data(training_log, start_step, end_step, batches_n):
    return {
        'log': training_log,
        'config': {
            'start_step': start_step,
            'end_step': end_step,
            'batches_n': batches_n,
            'epochs': FLAGS.epochs,
            'batch_size': FLAGS.batch_size,
            'learning_rate': FLAGS.learning_rate,
            'dropout': FLAGS.dropout,
            'l2_reg': FLAGS.l2_reg
        }
    }


def _save_log(log_data, model_folder):
    _assert_folder_exists(model_folder)
    start_step = log_data['config']['start_step']
    end_step = log_data['config']['end_step']
    file_name = 'training_log_' + str(start_step) + '_' + str(end_step) + '.p'
    file_path = os.path.join(model_folder, file_name)
    with open(file_path, 'wb') as f:
        pickle.dump(log_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print('Training log saved to: {}'.format(file_path))


def _plot_log(log_data, model_folder):

    config = log_data['config']
    training_log = np.array(log_data['log'])
    start_step = config['start_step']
    end_step = config['end_step']
    epochs = config['epochs']
    batches_n = config['batches_n']
    batch_size = config['batch_size']
    learning_rate = config['learning_rate']
    dropout = config['dropout']
    l2_reg = config['l2_reg']

    loss_log = training_log[:, 0]
    iou_log = training_log[:, 1]

    text = 'Loss: {:.3f}, IOU: {:.3f}'.format(loss_log[-1], iou_log[-1])

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    c1 = colors[0]
    c2 = colors[1]

    fig, ax1 = plt.subplots(figsize=(9, 6))

    ax1.set_xlabel('Batch')
    ax1.set_ylabel('Loss')

    x = np.arange(start_step + 1, end_step + 1, batches_n + 1)

    ax1.plot(x, loss_log, label='Loss', color=c1, marker='o')
    plt.xticks(x, x)

    ax2 = ax1.twinx()
    ax2.plot(x, iou_log, label='IOU', color=c2, marker='^')
    ax2.set_ylabel('IOU')

    handles, labels = ax1.get_legend_handles_labels()
    handles_2, labels_2 = ax2.get_legend_handles_labels()

    handles += handles_2
    labels += labels_2

    fig.legend(handles, labels, loc=(0.8, 0.6))
    fig.tight_layout()

    plt.title("(EP: {}, BS: {}, LR: {}, DO: {}, L2: {})".format(epochs, batch_size, learning_rate, dropout, l2_reg))

    fig.text(0.5, 0, text, verticalalignment='top', horizontalalignment='center', color='black', fontsize=10)

    fig.show()

    file_name = 'training_log_' + str(start_step) + '_' + str(end_step) + '.png'
    fig.savefig(os.path.join(model_folder, file_name), bbox_inches='tight')


def _checkpoint_exists(model_folder):
    return os.path.isfile(os.path.join(model_folder, 'checkpoint'))


def _model_checkpoint(model_folder):
    file_name = MODEL_NAME + MODEL_EXT
    return os.path.join(model_folder, file_name)


def _model_folder():
    model_folder = FLAGS.model_folder
    if model_folder is None:
        model_folder = os.path.join(
            MODEL_DIR, 'model_e' + str(FLAGS.epochs) + '_bs' + str(FLAGS.batch_size) + '_lr' + str(FLAGS.learning_rate)
            + '_do' + str(FLAGS.dropout) + '_l2' + str(FLAGS.l2_reg))
    return model_folder


def _summary_writer(sess, model_folder):
    model_folder_name = os.path.basename(model_folder)
    return tf.summary.FileWriter(os.path.join(LOGS_DIR, model_folder_name), graph=sess.graph)


def _conv_1x1(x, filters, name, regularizer=None):
    return tf.layers.conv2d(
        x,
        filters=filters,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding='same',
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
        kernel_regularizer=regularizer,
        name=name)


def _up_sample(x, filters, name, kernel_size, strides, regularizer=None):
    return tf.layers.conv2d_transpose(
        x,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding='same',
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
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

    # Scale layers (See optimized at-once architecture from the original implementation
    # of FCN-8s PASCAL at-once: https://github.com/shelhamer/fcn.berkeleyvision.org)
    layer3_scaled = tf.multiply(vgg_layer3_out, 0.0001, name='layer3_scaled')
    layer4_scaled = tf.multiply(vgg_layer4_out, 0.01, name='layer4_scaled')

    # 1x1 convolutions to the encoder layers
    layer3_1x1 = _conv_1x1(layer3_scaled, num_classes, 'layer3_1x1', regularizer=l2_reg)
    layer4_1x1 = _conv_1x1(layer4_scaled, num_classes, 'layer4_1x1', regularizer=l2_reg)
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
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    logits = tf.reshape(nn_last_layer, (-1, num_classes))

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)

    # Applies L2 regularization
    cross_entropy_loss = tf.reduce_mean(cross_entropy) + tf.losses.get_regularization_loss()

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=FLAGS.eps)

    global_step = tf.Variable(initial_value=0, trainable=False, name='global_step')

    train_op = optimizer.minimize(cross_entropy_loss, global_step=global_step)

    return logits, train_op, cross_entropy_loss, global_step


def iou_metric(model_output, labels, num_classes):
    logits_argmax = tf.argmax(tf.nn.softmax(model_output), axis=-1)
    labels_argmax = tf.argmax(labels, axis=-1)
    return tf.metrics.mean_iou(labels_argmax, logits_argmax, num_classes)


def train_nn(sess,
             global_step,
             epochs,
             batch_size,
             get_batches_fn,
             batches_n,
             train_op,
             cross_entropy_loss,
             iou_op,
             iou_mean,
             image_input,
             labels,
             keep_prob,
             learning_rate,
             save_model=False,
             tensorboard=False):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param iou_op: TF operation to update the iou metric
    :param iou_mean: TF tensor for the mean iou
    :param image_input: TF Placeholder for input images
    :param labels: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """

    #model_folder = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_folder = _model_folder()

    if save_model and _checkpoint_exists(model_folder):
        print('Checkpoint exists, restoring model from {}'.format(model_folder))
        _load_model(sess, model_folder)
    else:
        sess.run(tf.global_variables_initializer())

    sess.run(tf.local_variables_initializer())

    if save_model:
        saver = tf.train.Saver(max_to_keep=MODELS_LIMIT)

    if tensorboard:
        tf.summary.scalar('mean_iou', iou_mean)
        tf.summary.scalar('total_loss', cross_entropy_loss)
        summary = tf.summary.merge_all()
        train_writer = _summary_writer(sess, model_folder)

    training_log = []

    start = time.time()

    # Evaluate current step
    step = global_step.eval(session=sess)
    start_step = step

    print('Model folder: {}'.format(model_folder))
    print('Training (First batch: {}, Epochs: {}, Batch Size: {}, Learning Rate: {}, Dropout: {}, L2 Reg: {})'.format(
        step + 1, FLAGS.epochs, FLAGS.batch_size, FLAGS.learning_rate, FLAGS.dropout, FLAGS.l2_reg))

    for epoch in range(epochs):

        total_loss = 0
        curr_loss = 0
        total_iou = 0
        curr_iou = 0
        images_n = 0

        batches = tqdm(
            get_batches_fn(batch_size),
            desc='Epoch {}/{} (Batch: {}, Loss: N/A, IOU: N/A)'.format(epoch + 1, epochs, step + 1),
            unit='batches',
            total=batches_n)

        for batch_images, batch_labels in batches:

            feed_dict = {
                image_input: batch_images,
                labels: batch_labels,
                keep_prob: (1 - FLAGS.dropout),
                learning_rate: FLAGS.learning_rate
            }

            # Train
            _ = sess.run(train_op, feed_dict=feed_dict)

            images_n += len(batch_images)

            # Evaluate
            loss, _, iou = sess.run([cross_entropy_loss, iou_op, iou_mean],
                                    feed_dict={
                                        image_input: batch_images,
                                        labels: batch_labels,
                                        keep_prob: 1.0
                                    })

            step = global_step.eval(session=sess)

            total_loss += loss * len(batch_images)
            curr_loss = total_loss / images_n

            total_iou += iou * len(batch_images)
            curr_iou = total_iou / images_n

            # Saves metrics for tensorboard
            if tensorboard:
                training_summary = sess.run(
                    summary, feed_dict={
                        image_input: batch_images,
                        labels: batch_labels,
                        keep_prob: 1.0
                    })
                train_writer.add_summary(training_summary, global_step=step)

            batches.set_description('Epoch {}/{} (Batch: {}, Loss: {:.4f}, IOU: {:.4f})'.format(
                epoch + 1, epochs, step, curr_loss, curr_iou))

        epoch_loss = total_loss / images_n
        epoch_iou = total_iou / images_n

        training_log.append((epoch_loss, epoch_iou))

        if epoch % 5 == 0 and save_model:
            _save_model(sess, saver, model_folder, global_step)

    elapsed = time.time() - start

    print("Training Completed ({:.1f} s): Last batch: {}, Loss: {:.4f}, IOU: {:.4f}".format(
        elapsed, step, total_loss / images_n, total_iou / images_n))

    if save_model:
        _save_model(sess, saver, model_folder, global_step)
        log_data = _to_log_data(training_log, start_step, step, batches_n)
        _save_log(log_data, model_folder)
        _plot_log(log_data, model_folder)


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

    vgg_path = helper.maybe_download_pretrained_vgg(FLAGS.data_dir)

    with tf.Session(config=get_config()) as sess:
        image_input, keep_prob, layer3, layer4, layer7 = load_vgg(sess, vgg_path)
        model_output = layers(layer3, layer4, layer7, CLASSES_N)
        logits = tf.reshape(model_output, (-1, CLASSES_N))

        _load_model(sess, _model_folder())

        print('Processing image: {}'.format(file_path))
        name, image = helper.process_image(file_path, sess, logits, keep_prob, image_input, IMAGE_SHAPE)
        scipy.misc.imsave(os.path.join(FLAGS.runs_dir, name), image)


def process_video(file_path):

    if not os.path.isfile(file_path):
        raise ValueError('The file {} does not exist'.format(file_path))

    video_output = os.path.join(FLAGS.runs_dir, os.path.basename(file_path))

    cap = cv2.VideoCapture(file_path)
    frame_n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    #height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter(video_output + '.avi', fourcc, fps, IMAGE_SHAPE[::-1])

    vgg_path = helper.maybe_download_pretrained_vgg(FLAGS.data_dir)

    with tf.Session(config=get_config()) as sess:
        image_input, keep_prob, layer3, layer4, layer7 = load_vgg(sess, vgg_path)
        model_output = layers(layer3, layer4, layer7, CLASSES_N)
        logits = tf.reshape(model_output, (-1, CLASSES_N))

        _load_model(sess, _model_folder())
        for i in trange(frame_n, desc='Processing', unit='frames'):
            if cap.isOpened():
                ret, frame = cap.read()
                if ret == True:
                    frame = helper.process_image(frame, sess, logits, keep_prob, image_input, IMAGE_SHAPE)
                    out.write(frame)

        cap.release()
        out.release()


def run_testing():

    vgg_path = helper.maybe_download_pretrained_vgg(FLAGS.data_dir)

    with tf.Session(config=get_config()) as sess:
        image_input, keep_prob, layer3, layer4, layer7 = load_vgg(sess, vgg_path)
        model_output = layers(layer3, layer4, layer7, CLASSES_N)
        logits = tf.reshape(model_output, (-1, CLASSES_N))

        _load_model(sess, _model_folder())

        helper.save_inference_samples(FLAGS.runs_dir, FLAGS.data_dir, sess, IMAGE_SHAPE, logits, keep_prob, image_input)


def run():

    # Download pretrained vgg model
    vgg_path = helper.maybe_download_pretrained_vgg(FLAGS.data_dir)
    # Create function to get batches
    dataset_path = os.path.join(FLAGS.data_dir, 'data_road/training')
    get_batches_fn, samples_n = helper.gen_batch_function(dataset_path, IMAGE_SHAPE)

    batches_n = int(math.ceil(float(samples_n) / FLAGS.batch_size))

    with tf.Session(config=get_config()) as sess:

        labels = tf.placeholder(tf.float32, [None, None, None, CLASSES_N], 'input_labels')
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')

        image_input, keep_prob, layer3, layer4, layer7 = load_vgg(sess, vgg_path)
        model_output = layers(layer3, layer4, layer7, CLASSES_N)

        logits, train_op, cross_entropy_loss, global_step = optimize(model_output, labels, learning_rate, CLASSES_N)

        iou_mean, iou_op = iou_metric(model_output, labels, CLASSES_N)

        train_nn(sess, global_step, FLAGS.epochs, FLAGS.batch_size, get_batches_fn, batches_n, train_op,
                 cross_entropy_loss, iou_op, iou_mean, image_input, labels, keep_prob, learning_rate, True, True)

        helper.save_inference_samples(FLAGS.runs_dir, FLAGS.data_dir, sess, IMAGE_SHAPE, logits, keep_prob, image_input)


def main(_):
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
