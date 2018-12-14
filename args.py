import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_string('data_dir', './data', "The folder containing the training data; default ./data")
flags.DEFINE_string('runs_dir', './runs', "The folder containing the output; default ./runs")
flags.DEFINE_integer('epochs', 50, "The number of epochs; default 50")
flags.DEFINE_integer('batch_size', 15, "The batch size; default 15")
flags.DEFINE_float('learning_rate', 1e-3, "The learning rate; default 0.001")
flags.DEFINE_float('dropout', 0.5, 'The dropout probability; default 0.5')
flags.DEFINE_float('l2_reg', 1e-3, 'The l2 regularization amount; default 0.001')
flags.DEFINE_boolean('scale', True, 'True if scaling should be applied to layers 3 and 4 of VGG')
flags.DEFINE_integer('early_stopping', None, 'If provided will stop if the loss is not decreasing for the the given amount of epochs')
flags.DEFINE_integer('samples_limit', None, 'To limit the number of samples to train/test on')
flags.DEFINE_boolean('train', True, 'If true run training, otherwise runs prediction on testing images with existing model')
flags.DEFINE_string('image', None, 'Pass a single image to segment')
flags.DEFINE_string('video', None, 'Pass a video to segment')
flags.DEFINE_string('augment', None, 'If True runs augmentation on the dataset, exporting in the given folder')
flags.DEFINE_string('serialize', None, 'The folder where to save the model as protobuffer')
flags.DEFINE_boolean('cpu', False, 'Force the use of CPU')
flags.DEFINE_string('model_folder', None, 'Specific model folder path to restore')
flags.DEFINE_boolean('tests', True, 'True if the tests should be run')
flags.DEFINE_integer('seed', None, 'Seed for reproducibility')