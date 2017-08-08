"""Builds the ocr network.

Summary of available functions:

 # Compute input images and labels for training. If you would like to run
 # evaluations, use inputs() instead.
 inputs, labels = distorted_inputs()

 # Compute inference on the model inputs to make a prediction.
 predictions = inference(inputs)

 # Compute the total loss of the prediction with respect to the labels.
 loss = loss(predictions, labels)

 # Create a graph to run one step of training with respect to the loss.
 train_op = train(loss, global_step)
"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys
import tarfile

from six.moves import urllib
import tensorflow as tf

import ocr_input

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('eval_batch_size', 8,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('train_data_dir', os.path.abspath(os.path.join("data", "tfrecords_train")),
                            """Path to train data directory.""")
tf.app.flags.DEFINE_string('eval_data_dir', os.path.abspath(os.path.join("data", "tfrecords_test")),
                            """Path to train data directory.""")

# Global constants describing the ocr data set.
CONV1_DEPTH = 32
CONV2_DEPTH = 48
CONV3_DEPTH = 64
CONV_FC_OUTPUT = 32
LSTM_HIDDEN_SIZE = 256 #384
NUM_LSTM_LAYERS = 2

NUM_CLASSES = ocr_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = ocr_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = ocr_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL


# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 20.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.0001       # Initial learning rate. 0.00003

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

def _activation_summary(x):
  """Helper to create summaries for activations.

  Creates a summary that provides a histogram of activations.
  Creates a summary that measures the sparsity of activations.

  Args:
    x: Tensor
  Returns:
    nothing
  """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.summary.histogram(tensor_name + '/activations', x)
  tf.summary.scalar(tensor_name + '/sparsity',
                                       tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    var = tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32)
  return var


def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

  Returns:
    Variable Tensor
  """
  var = _variable_on_cpu(
      name,
      shape,
      tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32))
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var


def distorted_inputs():
  """Construct distorted input for CIFAR training using the Reader ops.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.

  Raises:
    ValueError: If no data_dir
  """
  with tf.variable_scope('distorted_inputs'):
    if not FLAGS.train_data_dir:
      raise ValueError('Please supply a data_dir')
    data_dir = FLAGS.train_data_dir
    images, labels, seq_lengths = ocr_input.distorted_inputs(data_dir=data_dir,
                                                    batch_size=FLAGS.batch_size)  
    # seq_lengths = tf.Print(seq_lengths, [seq_lengths], "seq_lengths")
    return images, labels, seq_lengths


def inputs():
  """Construct input for CIFAR evaluation using the Reader ops.

  Args:
    eval_data: bool, indicating if one should use the train or eval data set.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.

  Raises:
    ValueError: If no data_dir
  """
  with tf.variable_scope('inputs'):
    if not FLAGS.eval_data_dir:
        raise ValueError('Please supply a data_dir')
    data_dir = FLAGS.eval_data_dir
    images, labels, seq_lengths = ocr_input.inputs(data_dir=data_dir, batch_size=FLAGS.eval_batch_size)
    return images, labels, seq_lengths

# Utility functions
def conv_weight_variable(shape):
    initial = tf.contrib.layers.xavier_initializer_conv2d()
    return tf.get_variable("w", shape=shape, initializer=initial)

def weight_variable(shape):
    initial = tf.contrib.layers.xavier_initializer()
    return tf.get_variable("w", shape=shape, initializer=initial)

def bias_variable(shape):
    initial = tf.zeros_initializer()
    return tf.get_variable("b", shape=shape, initializer=initial)


def conv2d(x, W, stride=(1, 1), padding='SAME'):
    return tf.nn.conv2d(x, W, strides=[1, stride[0], stride[1], 1],
                        padding=padding)


def max_pool(x, ksize=(2, 2), stride=(2, 2)):
    return tf.nn.max_pool(x, ksize=[1, ksize[0], ksize[1], 1],
                          strides=[1, stride[0], stride[1], 1], padding='SAME')


def avg_pool(x, ksize=(2, 2), stride=(2, 2)):
    return tf.nn.avg_pool(x, ksize=[1, ksize[0], ksize[1], 1],
                          strides=[1, stride[0], stride[1], 1], padding='SAME')

def convolutional_layers(images, batch_size, train):
    """
    Get the convolutional layers of the model.
    """

    
    with tf.variable_scope('convolutions'):
      # First layer
      with tf.variable_scope('conv1') as scope:
        W_conv1 = conv_weight_variable([3, 3, ocr_input.IMAGE_DEPTH, CONV1_DEPTH])
        b_conv1 = bias_variable([CONV1_DEPTH])
        h_conv1 = conv2d(images, W_conv1) + b_conv1
        h_bn1 = tf.contrib.layers.batch_norm(h_conv1,
                                            center=True, scale=True,
                                            is_training=train,
                                            scope='bn')
        h_pool1 = max_pool(h_bn1, ksize=(2, 2), stride=(2, 2))
        h_relu1 = tf.nn.relu(h_pool1, name=scope.name)
        _activation_summary(h_relu1)

      # Second layer
      with tf.variable_scope('conv2') as scope:
        W_conv2 = conv_weight_variable([3, 3, CONV1_DEPTH, CONV2_DEPTH])
        b_conv2 = bias_variable([CONV2_DEPTH])
        h_conv2 = conv2d(h_relu1, W_conv2) + b_conv2
        h_bn2 = tf.contrib.layers.batch_norm(h_conv2,
                                            center=True, scale=True,
                                            is_training=train,
                                            scope='bn')
        h_pool2 = max_pool(h_bn2, ksize=(2, 2), stride=(2, 2))
        h_relu2 = tf.nn.relu(h_pool2, name=scope.name)
        _activation_summary(h_relu2)

        # Second layer
      with tf.variable_scope('conv3') as scope:
          W_conv3 = conv_weight_variable([3, 3, CONV2_DEPTH, CONV3_DEPTH])
          b_conv3 = bias_variable([CONV3_DEPTH])
          h_conv3 = conv2d(h_relu2, W_conv3) + b_conv3
          h_bn3 = tf.contrib.layers.batch_norm(h_conv3,
                                               center=True, scale=True,
                                               is_training=train,
                                               scope='bn')
          h_pool3 = max_pool(h_bn3, ksize=(2, 2), stride=(2, 2))
          h_relu3 = tf.nn.relu(h_pool3, name=scope.name)
          _activation_summary(h_relu3)

      """
      # Third layer timestep-wise classifier and dimensionality reductioner
      with tf.variable_scope('conv3_dim_redux'):
        W_conv3 = weight_variable([1, 8, 32, CONV_FC_OUTPUT])
        b_conv3 = bias_variable([CONV_FC_OUTPUT])
        h_conv3 = conv2d(h_pool2, W_conv3, stride=(1, 1), padding='VALID') + b_conv3
        # h_bn3 = tf.contrib.layers.batch_norm(h_conv3, 
        #                                     center=True, scale=True, 
        #                                     is_training=True,
        #                                     scope='bn')        

      print()
      print("h_relu3")
      print(h_relu3)
      """

      
      with tf.variable_scope('dim_redux') as scope:
        conv_out_shape = tf.shape(h_pool3)
        print("Conv_out_shape:", str(conv_out_shape))
        # w_fc1 = weight_variable([conv_out_shape[2] * 32, CONV_FC_OUTPUT])
        w_fc1 = weight_variable([(ocr_input.IMAGE_HEIGHT / 2 / 2 / 2) * CONV3_DEPTH, CONV_FC_OUTPUT])
        b_fc1 = bias_variable([CONV_FC_OUTPUT])        
        conv_layer_flat = tf.reshape(h_relu3, [-1, conv_out_shape[2] * CONV3_DEPTH])
        features = tf.matmul(conv_layer_flat, w_fc1) + b_fc1
        h_bn3 = tf.contrib.layers.batch_norm(features,
                                            center=True, scale=True,
                                            is_training=train,
                                            scope='bn')
        features = tf.nn.relu(h_bn3)
        features = tf.reshape(features, [batch_size, conv_out_shape[1], CONV_FC_OUTPUT])
        _activation_summary(features)
 
      timesteps = tf.fill([batch_size], conv_out_shape[1])       
      return features, timesteps

def stack_bidirectional_dynamic_rnn(cells_fw,
                                    cells_bw,
                                    inputs,
                                    initial_states_fw=None,
                                    initial_states_bw=None,
                                    dtype=None,
                                    sequence_length=None,
                                    scope=None):
  """Creates a dynamic bidirectional recurrent neural network.
 
  Stacks several bidirectional rnn layers. The combined forward and backward
  layer outputs are used as input of the next layer. tf.bidirectional_rnn
  does not allow to share forward and backward information between layers.
  The input_size of the first forward and backward cells must match.
  The initial state for both directions is zero and no intermediate states
  are returned.
 
  Args:
    cells_fw: List of instances of RNNCell, one per layer,
      to be used for forward direction.
    cells_bw: List of instances of RNNCell, one per layer,
      to be used for backward direction.
    inputs: A length T list of inputs, each a tensor of shape
      [batch_size, input_size], or a nested tuple of such elements.
    initial_states_fw: (optional) A list of the initial states (one per layer)
      for the forward RNN.
      Each tensor must has an appropriate type and shape
      `[batch_size, cell_fw.state_size]`.
    initial_states_bw: (optional) Same as for `initial_states_fw`, but using
      the corresponding properties of `cells_bw`.
    dtype: (optional) The data type for the initial state.  Required if
      either of the initial states are not provided.
    sequence_length: (optional) An int32/int64 vector, size `[batch_size]`,
      containing the actual lengths for each of the sequences.
    scope: VariableScope for the created subgraph; defaults to None.
 
  Returns:
    A tuple (outputs, output_state_fw, output_state_bw) where:
      outputs: Output `Tensor` shaped:
        `batch_size, max_time, layers_output]`. Where layers_output
        are depth-concatenated forward and backward outputs.
      output_states_fw is the final states, one tensor per layer,
        of the forward rnn.
      output_states_bw is the final states, one tensor per layer,
        of the backward rnn.
 
  Raises:
    TypeError: If `cell_fw` or `cell_bw` is not an instance of `RNNCell`.
    ValueError: If inputs is `None`, not a list or an empty list.
  """
  if not cells_fw:
    raise ValueError("Must specify at least one fw cell for BidirectionalRNN.")
  if not cells_bw:
    raise ValueError("Must specify at least one bw cell for BidirectionalRNN.")
  if not isinstance(cells_fw, list):
    raise ValueError("cells_fw must be a list of RNNCells (one per layer).")
  if not isinstance(cells_bw, list):
    raise ValueError("cells_bw must be a list of RNNCells (one per layer).")
  if len(cells_fw) != len(cells_bw):
    raise ValueError("Forward and Backward cells must have the same depth.")
  if initial_states_fw is not None and (not isinstance(cells_fw, list) or
                                        len(cells_fw) != len(cells_fw)):
    raise ValueError(
        "initial_states_fw must be a list of state tensors (one per layer).")
  if initial_states_bw is not None and (not isinstance(cells_bw, list) or
                                        len(cells_bw) != len(cells_bw)):
    raise ValueError(
        "initial_states_bw must be a list of state tensors (one per layer).")
 
  states_fw = []
  states_bw = []
  prev_layer = inputs
 
  with vs.variable_scope(scope or "StackRNN"):
    for i, (cell_fw, cell_bw) in enumerate(zip(cells_fw, cells_bw)):
      initial_state_fw = None
      initial_state_bw = None
      if initial_states_fw:
        initial_state_fw = initial_states_fw[i]
      if initial_states_bw:
        initial_state_bw = initial_states_bw[i]
 
      with vs.variable_scope("Layer%d" % i):
        outputs, (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
            cell_fw,
            cell_bw,
            prev_layer,
            initial_state_fw=initial_state_fw,
            initial_state_bw=initial_state_bw,
            sequence_length=sequence_length,
            dtype=dtype)
        # Concat the outputs to create the new input.
        prev_layer = tf.concat(2, outputs)
      states_fw.append(state_fw)
      states_bw.append(state_bw)
 
  return prev_layer, tuple(states_fw), tuple(states_bw)

#features - features extracted from image using CNN
def get_lstm_layers(features, timesteps, batch_size):
    with tf.variable_scope('RNN'):
      # Has size [batch_size, max_stepsize, num_features], but the
      # batch_size and max_stepsize can vary along each step
      #tf.placeholder(tf.float32, [None, None, ocr_input.IMAGE_HEIGHT])
      inputs = features
      shape = tf.shape(features)
      batch_size, max_timesteps = shape[0], shape[1]      

      # Defining the cell
      # Can be:
      #   tf.nn.rnn_cell.RNNCell
      #   tf.nn.rnn_cell.GRUCell
      cell = tf.contrib.rnn.core_rnn_cell.LSTMCell(LSTM_HIDDEN_SIZE, state_is_tuple=True)

      # Stacking rnn cells
      stack = tf.contrib.rnn.core_rnn_cell.MultiRNNCell([cell] * NUM_LSTM_LAYERS,
                                          state_is_tuple=True)

      # The second output is the last state and we will no use that
      outputs, _ = tf.nn.dynamic_rnn(stack, inputs, timesteps, dtype=tf.float32)          

      # Reshaping to apply the same weights over the timesteps
      outputs = tf.reshape(outputs, [-1, LSTM_HIDDEN_SIZE])
      # outputs = tf.Print(outputs, [outputs], "Outputs")

      with tf.variable_scope('logits'):
        w = tf.Variable(tf.truncated_normal([LSTM_HIDDEN_SIZE,
                                           NUM_CLASSES],
                                          stddev=0.1), name="w")
        b = tf.Variable(tf.constant(0., shape=[NUM_CLASSES]), name="b")

        # Doing the affine projection
        logits = tf.matmul(outputs, w) + b

        # Reshaping back to the original shape
        logits = tf.reshape(logits, [batch_size, -1, NUM_CLASSES])

        logits = tf.transpose(logits, [1, 0, 2], name="out_logits")

      return logits

def create_ctc_loss(logits, labels, timesteps, label_seq_lengths):
  with tf.variable_scope('CTC_Loss'):
    print()
    print("Labels shape")
    print(labels)
    print()
    print("Logits shape")
    print(logits)
    print()
    print("Labels len  shape")
    print(label_seq_lengths)

    # logits = tf.Print(logits, [logits], "Logits")
    ctc_loss = tf.nn.ctc_loss(labels, 
                   logits, 
                   timesteps)

    cost = tf.reduce_mean(ctc_loss, name='ctc')

    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).
    return cost

def create_label_error_rate(logits, labels, timesteps):
  with tf.variable_scope('LER'):
    decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, timesteps)
    decoded = tf.cast(decoded[0], tf.int32)    
    edit_dist = tf.edit_distance(decoded, labels)
    ler = tf.reduce_mean(edit_dist)
    tf.summary.scalar('label_error_rate', ler)
    return ler

def check_decoder(logits, labels, timesteps):
  with tf.variable_scope('check_decoder'):
    decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, timesteps)
    decoded = tf.cast(decoded[0], tf.int32)
    decoded = tf.sparse_tensor_to_dense(decoded)
    # decoded = tf.Print(decoded, [decoded], "Decoded")    
    return decoded

def inference(images, batch_size, train):
  """Build the ocr model.

  Args:
    images: Images returned from distorted_inputs() or inputs().

  Returns:
    Logits.
  """
  features, timesteps = convolutional_layers(images, batch_size, train)
  logits = get_lstm_layers(features, timesteps, batch_size)
  return logits, timesteps
  

  # The total loss is defined as the cross entropy loss plus all of the weight
  # decay terms (L2 loss).
  # return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _add_loss_summaries(total_loss):
  """Add summaries for losses in ocr model.

  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.

  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.summary.scalar(l.op.name + ' (raw)', l)
    tf.summary.scalar(l.op.name, loss_averages.average(l))

  return loss_averages_op

def train_simple(total_loss, global_step):
  with tf.variable_scope('train_op'):
    # Variables that affect learning rate.
    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                    global_step,
                                    decay_steps,
                                    LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)
    tf.summary.scalar('learning_rate', lr)

    # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    # with tf.control_dependencies(update_ops):
        # opt = tf.train.MomentumOptimizer(lr, 0.9).minimize(total_loss, global_step=global_step)
    opt = tf.train.AdamOptimizer(lr).minimize(total_loss, global_step=global_step)

    tf.summary.scalar(total_loss.op.name + ' (raw)', total_loss)
  return opt, lr

def train(total_loss, global_step):
  """Train ocr model.

  Create an optimizer and apply to all trainable variables. Add moving
  average for all trainable variables.

  Args:
    total_loss: Total loss from loss().
    global_step: Integer Variable counting the number of training steps
      processed.
  Returns:
    train_op: op for training.
  """
  # Variables that affect learning rate.
  num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
  decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

  # Decay the learning rate exponentially based on the number of steps.
  lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                  global_step,
                                  decay_steps,
                                  LEARNING_RATE_DECAY_FACTOR,
                                  staircase=True)
  tf.summary.scalar('learning_rate', lr)

  # Generate moving averages of all losses and associated summaries.
  loss_averages_op = _add_loss_summaries(total_loss)

  # Compute gradients.
  with tf.control_dependencies([loss_averages_op]):
    opt = tf.train.AdamOptimizer(lr)
    grads = opt.compute_gradients(total_loss)

  # Apply gradients.
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

  # Add histograms for trainable variables.
  for var in tf.trainable_variables():
    tf.summary.histogram(var.op.name, var)

  # Add histograms for gradients.
  for grad, var in grads:
    if grad is not None:
      tf.summary.histogram(var.op.name + '/gradients', grad)

  # Track the moving averages of all trainable variables.
  variable_averages = tf.train.ExponentialMovingAverage(
      MOVING_AVERAGE_DECAY, global_step)
  variables_averages_op = variable_averages.apply(tf.trainable_variables())

  with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
    train_op = tf.no_op(name='train')

  return train_op