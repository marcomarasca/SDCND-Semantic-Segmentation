import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_string('data_dir', './data', "The folder containing the training data; default ./data")
flags.DEFINE_string('runs_dir', './runs', "The folder containing the output; default ./runs")
flags.DEFINE_integer('epochs', 50, "The number of epochs; default 50")
flags.DEFINE_integer('batch_size', 5, "The batch size; default 1")
flags.DEFINE_float('learning_rate', 0.0001, "The learning rate; default 0.0001")
flags.DEFINE_float('dropout', 0.5, 'The dropout probability; default 0.5')
flags.DEFINE_float('l2_beta', 0.001, 'The l2 regularization beta; default 0.001')