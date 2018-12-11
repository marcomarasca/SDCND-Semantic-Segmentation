import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_string('data_dir', './data', "The folder containing the training data; default ./data")
flags.DEFINE_string('runs_dir', './runs', "The folder containing the output; default ./runs")
flags.DEFINE_integer('epochs', 50, "The number of epochs; default 50")
flags.DEFINE_integer('batch_size', 5, "The batch size; default 1")
flags.DEFINE_float('learning_rate', 1e-4, "The learning rate; default 0.0001")
flags.DEFINE_float('dropout', 0.5, 'The dropout probability; default 0.5')
flags.DEFINE_float('w_std', 1e-3, 'The std deviation for the decoder kernel initializer; default 0.001')
flags.DEFINE_float('l2_reg', 1e-3, 'The l2 regularization amount; default 0.001')
flags.DEFINE_boolean('cpu', False, 'Force the use of CPU')
flags.DEFINE_boolean('restore', False, 'Restore training from checkpoint')