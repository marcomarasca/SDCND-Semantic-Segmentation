'''
You should not edit helper.py as part of your submission.

This file is used primarily to download vgg if it has not yet been,
give you the progress of the download, get batches for your training,
as well as around generating and saving the image outputs.
'''
from args import FLAGS

import re
import random
import os.path
import scipy.misc
import shutil
import zipfile
import time
import pickle
import numpy as np
import tensorflow as tf
import matplotlib as mpl
# For plotting without a screen
mpl.use('Agg')
import matplotlib.pyplot as plt

from glob import glob
from urllib.request import urlretrieve
from tqdm import tqdm

MODEL_DIR = 'models'
MODEL_NAME = 'fcn-vgg16'
MODEL_EXT = '.ckpt'
LOGS_DIR = 'logs'

if not os.path.isdir(MODEL_DIR):
    os.makedirs(MODEL_DIR)

if not os.path.isdir(LOGS_DIR):
    os.makedirs(LOGS_DIR)


class DLProgress(tqdm):
    """
	Report download progress to the terminal.
	:param tqdm: Information fed to the tqdm library to estimate progress.
	"""
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        """
		Store necessary information for tracking progress.
		:param block_num: current block of the download
		:param block_size: size of current block
		:param total_size: total download size, if known
		"""
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)  # Updates progress
        self.last_block = block_num


def maybe_download_pretrained_vgg(data_dir):
    """
	Download and extract pretrained vgg model if it doesn't exist
	:param data_dir: Directory to download the model to
	"""
    vgg_filename = 'vgg.zip'
    vgg_path = os.path.join(data_dir, 'vgg')
    vgg_files = [
        os.path.join(vgg_path, 'variables/variables.data-00000-of-00001'),
        os.path.join(vgg_path, 'variables/variables.index'),
        os.path.join(vgg_path, 'saved_model.pb')
    ]

    missing_vgg_files = [vgg_file for vgg_file in vgg_files if not os.path.exists(vgg_file)]
    if missing_vgg_files:
        # Clean vgg dir
        if os.path.exists(vgg_path):
            shutil.rmtree(vgg_path)
        os.makedirs(vgg_path)

        # Download vgg
        print('Downloading pre-trained vgg model...')
        with DLProgress(unit='B', unit_scale=True, miniters=1) as pbar:
            urlretrieve('https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip',
                        os.path.join(vgg_path, vgg_filename), pbar.hook)

        # Extract vgg
        print('Extracting model...')
        zip_ref = zipfile.ZipFile(os.path.join(vgg_path, vgg_filename), 'r')
        zip_ref.extractall(data_dir)
        zip_ref.close()

        # Remove zip file to save space
        os.remove(os.path.join(vgg_path, vgg_filename))

    return vgg_path


def _assert_folder_exists(folder):
    if not os.path.isdir(folder):
        raise ValueError('The folder {} does not exist'.format(folder))


def _model_checkpoint(model_folder):
    file_name = MODEL_NAME + MODEL_EXT
    return os.path.join(model_folder, file_name)


def save_model(sess, saver, model_folder, global_step):
    model_path = _model_checkpoint(model_folder)
    save_path = saver.save(sess, model_path, global_step=global_step)
    print('Model saved in path: {}'.format(save_path))


def load_model(sess, model_folder):
    _assert_folder_exists(model_folder)
    saver = tf.train.Saver()
    # Restore the latest checkpoint
    model_path = tf.train.latest_checkpoint(model_folder)
    saver.restore(sess, model_path)
    print('Model restored from path: {}'.format(model_path))


def to_log_data(training_log, start_step, end_step, batches_n):
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
            'l2_reg': FLAGS.l2_reg,
            'eps': FLAGS.eps,
            'scale': FLAGS.scale
        }
    }


def summary_writer(sess, model_folder):
    model_folder_name = os.path.basename(model_folder)
    return tf.summary.FileWriter(os.path.join(LOGS_DIR, model_folder_name), graph=sess.graph)


def image_summary_batch(data_folder, image_shape, image_n):
    batch_fn, samples_n = gen_batch_function(data_folder, image_shape)
    return next(batch_fn(image_n))


def save_log(log_data, model_folder):
    _assert_folder_exists(model_folder)
    start_step = log_data['config']['start_step']
    end_step = log_data['config']['end_step']
    file_name = 'training_log_' + str(start_step) + '_' + str(end_step) + '.p'
    file_path = os.path.join(model_folder, file_name)
    with open(file_path, 'wb') as f:
        pickle.dump(log_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print('Training log saved to: {}'.format(file_path))


def plot_log(log_data, model_folder):

    config = log_data['config']
    training_log = np.array(log_data['log'])
    start_step = config['start_step']
    end_step = config['end_step']
    epochs = config['epochs']
    batches_n = config['batches_n']
    batch_size = config['batch_size']
    learning_rate = config['learning_rate']
    dropout = config['dropout']
    eps = config['eps']
    l2_reg = config['l2_reg']
    scale = config['scale']

    loss_log = training_log[:, 0]
    acc_log = training_log[:, 1]
    iou_log = training_log[:, 2]

    text = 'Loss: {:.4f}, Acc: {:.4f}, IoU: {:.4f}'.format(loss_log[-1], acc_log[-1], iou_log[-1])

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    c1 = colors[0]
    c2 = colors[1]
    c3 = colors[3]

    fig, ax1 = plt.subplots(figsize=(10, 7))

    ax1.set_xlabel('Step (Epochs)')
    ax1.set_ylabel('Loss/Accuracy')

    x = np.arange(min(start_step + batches_n, end_step), end_step + 1, batches_n)

    ax1.plot(x, loss_log, label='Loss', color=c1, marker='.')
    ax1.plot(x, acc_log, label='Accuracy', color=c2, marker='.')
    plt.xticks(x, x, rotation=(0 if len(x) < 20 else 80))

    ax2 = ax1.twinx()
    ax2.plot(x, iou_log, label='IOU', color=c3, marker='s')
    ax2.set_ylabel('IOU')

    handles, labels = ax1.get_legend_handles_labels()
    handles_2, labels_2 = ax2.get_legend_handles_labels()

    handles += handles_2
    labels += labels_2

    fig.legend(handles, labels, loc=(0.7, 0.5))
    fig.tight_layout()

    plt.title("(EP: {}, BS: {}, LR: {}, DO: {}, L2: {}, EPS: {}, S: {})".format(
        epochs, batch_size, learning_rate, dropout, l2_reg, eps, 'ON' if scale else 'OFF'))

    fig.text(0.5, 0, text, verticalalignment='top', horizontalalignment='center', color='black', fontsize=10)

    file_name = 'training_log_' + str(start_step) + '_' + str(end_step) + '.png'
    fig.savefig(os.path.join(model_folder, file_name), bbox_inches='tight')


def checkpoint_exists(model_folder):
    return os.path.isfile(os.path.join(model_folder, 'checkpoint'))


def model_folder():
    model_folder = FLAGS.model_folder
    if model_folder is None:
        file_name = 'm_e=' + str(FLAGS.epochs) + '_bs=' + str(FLAGS.batch_size) + '_lr=' + str(
            FLAGS.learning_rate) + '_do=' + str(FLAGS.dropout) + '_l2=' + str(FLAGS.l2_reg) + '_eps=' + str(
                FLAGS.eps) + '_scale=' + ('on' if FLAGS.scale else 'off')
        model_folder = os.path.join(MODEL_DIR, file_name)
    return model_folder


def config_tensor():
    return tf.stack([
        tf.convert_to_tensor(['epochs', str(FLAGS.epochs)]),
        tf.convert_to_tensor(['batch_size', str(FLAGS.batch_size)]),
        tf.convert_to_tensor(['learning_rate', str(FLAGS.learning_rate)]),
        tf.convert_to_tensor(['dropout', str(FLAGS.dropout)]),
        tf.convert_to_tensor(['l2_reg', str(FLAGS.l2_reg)]),
        tf.convert_to_tensor(['eps', str(FLAGS.eps)]),
        tf.convert_to_tensor(['scale', 'ON' if FLAGS.scale else 'OFF'])
    ])


def limit_samples(paths):
    if FLAGS.samples_limit is not None:
        paths = paths[:FLAGS.samples_limit]
    return paths


def gen_batch_function(data_folder, image_shape):
    """
	Generate function to create batches of training data
	:param data_folder: Path to folder that contains all the datasets
	:param image_shape: Tuple - Shape of image
	:return:
	"""
    # Grab image and label paths
    image_paths = glob(os.path.join(data_folder, 'image_2', '*.png'))
    label_paths = {
        re.sub(r'_(lane|road)_', '_', os.path.basename(path)): path
        for path in glob(os.path.join(data_folder, 'gt_image_2', '*_road_*.png'))
    }
    background_color = np.array([255, 0, 0])

    image_paths = limit_samples(image_paths)

    samples_n = len(image_paths)

    rnd = random.Random(FLAGS.seed)

    def get_batches_fn(batch_size):
        """
		Create batches of training data
		:param batch_size: Batch Size
		:return: Batches of training data
		"""
        # Shuffle training data
        rnd.shuffle(image_paths)
        # Loop through batches and grab images, yielding each batch
        for batch_i in range(0, samples_n, batch_size):
            images = []
            gt_images = []
            for image_file in image_paths[batch_i:batch_i + batch_size]:
                gt_image_file = label_paths[os.path.basename(image_file)]
                # Re-size to image_shape
                image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)
                gt_image = scipy.misc.imresize(scipy.misc.imread(gt_image_file), image_shape)

                # Create "one-hot-like" labels by class
                gt_bg = np.all(gt_image == background_color, axis=2)
                gt_bg = gt_bg.reshape(*gt_bg.shape, 1)
                gt_image = np.concatenate((gt_bg, np.invert(gt_bg)), axis=2)

                images.append(image)
                gt_images.append(gt_image)

            yield np.array(images), np.array(gt_images)

    return get_batches_fn, samples_n


def process_image(input_image, sess, logits, keep_prob, image_pl, image_shape):
    """
	Process a single image
	:param input_image: The image 
	:param sess: TF session
	:param logits: TF Tensor for the logits
	:param keep_prob: TF Placeholder for the dropout keep probability
	:param image_pl: TF Placeholder for the image placeholder
	:param data_folder: Path to the folder that contains the datasets
	:param image_shape: Tuple - Shape of image
	:return: A segmented image
	"""
    image = scipy.misc.imresize(input_image, image_shape)
    # Run inference
    im_softmax = sess.run([tf.nn.softmax(logits)], {keep_prob: 1.0, image_pl: [image]})
    # Splice out second column (road), reshape output back to image_shape
    im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
    # If road softmax > 0.5, prediction is road
    segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
    # Create mask based on segmentation to apply to original image
    mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
    mask = scipy.misc.toimage(mask, mode="RGBA")
    street_im = scipy.misc.toimage(image)
    street_im.paste(mask, box=None, mask=mask)

    return np.array(street_im)


def process_image_file(file_path, sess, logits, keep_prob, image_pl, image_shape):
    """
	Process a single image from the given path
	:param file_path: The image path
	:param sess: TF session
	:param logits: TF Tensor for the logits
	:param keep_prob: TF Placeholder for the dropout keep probability
	:param image_pl: TF Placeholder for the image placeholder
	:param data_folder: Path to the folder that contains the datasets
	:param image_shape: Tuple - Shape of image
	:return: A pair with the file name and the segmented image
	"""
    street_im = process_image(scipy.misc.imread(file_path), sess, logits, keep_prob, image_pl, image_shape)

    return os.path.basename(file_path), street_im


def gen_test_output(sess, logits, keep_prob, image_pl, data_folder, image_shape):
    """
	Generate test output using the test images
	:param sess: TF session
	:param logits: TF Tensor for the logits
	:param keep_prob: TF Placeholder for the dropout keep probability
	:param image_pl: TF Placeholder for the image placeholder
	:param data_folder: Path to the folder that contains the datasets
	:param image_shape: Tuple - Shape of image
	:return: Output for for each test image
	"""

    image_paths = glob(os.path.join(data_folder, 'image_2', '*.png'))

    image_paths = limit_samples(image_paths)

    for image_file in tqdm(image_paths, desc='Processing: ', unit='images', total=len(image_paths)):
        yield process_image_file(image_file, sess, logits, keep_prob, image_pl, image_shape)


def save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image):
    """
	Save test images with semantic masks of lane predictions to runs_dir.
	:param runs_dir: Directory to save output images
	:param data_dir: Path to the directory that contains the datasets
	:param sess: TF session
	:param image_shape: Tuple - Shape of image
	:param logits: TF Tensor for the logits
	:param keep_prob: TF Placeholder for the dropout keep probability
	:param input_image: TF Placeholder for the image placeholder
	"""
    # Make folder for current run
    output_dir = os.path.join(runs_dir, str(time.time()))
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Run NN on test images and save them to HD
    print('Training Finished. Saving test images to: {}'.format(output_dir))
    image_outputs = gen_test_output(sess, logits, keep_prob, input_image, os.path.join(
        data_dir, 'data_road', 'testing'), image_shape)
    for name, image in image_outputs:
        scipy.misc.imsave(os.path.join(output_dir, name), image)
