"""A binary to train ocr using a single GPU."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time
import tensorflow as tf
import ocr
import ocr_input
import os

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', 'train_logs',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', True,
                            """Whether to log device placement.""")


def train():
    """Train ocr for a number of steps."""
    with tf.Graph().as_default():
        global_step = tf.contrib.framework.get_or_create_global_step()

        # Get images and labels for ocr.
        print("Preparing input")
        # with tf.device('/cpu:0'):
        images, labels, seq_lengths = ocr.distorted_inputs()

        # Build a Graph that computes the logits predictions from the
        # inference model.
        print("Building graph")
        logits, timesteps = ocr.inference(images, FLAGS.batch_size, train=True)

        # Calculate loss.
        print("Creating loss")        
        loss = ocr.create_ctc_loss(logits, labels, timesteps, seq_lengths)

        print("Creating LER")
        ler = ocr.create_label_error_rate(logits, labels, timesteps)

        print("Creating decoder")
        decoded = ocr.check_decoder(logits, labels, timesteps)

        # Build a Graph that trains the model with one batch of examples and
        # updates the model parameters.
        print("Creating train OP")
        train_op, lr = ocr.train_simple(loss, global_step)

        print("Creating init OP")
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())

        sess = tf.Session()

        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        
        train_writer = tf.summary.FileWriter(FLAGS.train_dir,
                                             sess.graph)
        saver = tf.train.Saver()
        summary_op = tf.summary.merge_all()

        print("Starting training")
        print_every_n = 1000
        start_time = time.time()
        mean_ler = 0
        while not coord.should_stop():
            try:                
                _, loss_res, lr_res, ler_res, summary_op_result, global_step_result, decoded_res = sess.run([train_op, loss, lr, ler, summary_op, global_step, decoded])
                mean_ler += ler_res
                if global_step_result % print_every_n == 0 or global_step_result == 1:
                    mean_steps_time = (time.time() - start_time) / print_every_n
                    mean_ler = mean_ler / print_every_n
                    status_string = "Step: {} Loss: {:.4f} LR: {:.6f} LER: {:.4f} Step time: {:.3f} sec"
                    print(status_string.format(global_step_result, loss_res, lr_res, ler_res, mean_steps_time))                    
                    # print("Decoded:")
                    # print(str(decoded_res))
                    # print("Timesteps:" + str(timesteps_res))
                    train_writer.add_summary(summary_op_result, global_step=global_step_result)
                    saver.save(sess, os.path.join(FLAGS.train_dir, 'checkpoint'), global_step=global_step)
                    start_time = time.time()
                    mean_ler = 0

                # images_res = sess.run(images)
                # print(images_res)                
                # for img in images_res:
                #     cv2.imshow("img", img)
                #     cv2.waitKey(0)
            except Exception as e:
                print(e)
                coord.request_stop(e)

            # class _LoggerHook(tf.train.SessionRunHook):
            # """Logs loss and runtime."""
            #
            # def begin(self):
            # self._step = -1
            #
            # def before_run(self, run_context):
            # self._step += 1
            # self._start_time = time.time()
            # return tf.train.SessionRunArgs(loss)  # Asks for loss value.
            #
            # def after_run(self, run_context, run_values):
            # duration = time.time() - self._start_time
            # loss_value = run_values.results
            # if self._step % 10 == 0:
            # num_examples_per_step = FLAGS.batch_size
            # examples_per_sec = num_examples_per_step / duration
            # sec_per_batch = float(duration)
            #
            # format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
            # 'sec/batch)')
            # print (format_str % (datetime.now(), self._step, loss_value,
            #  examples_per_sec, sec_per_batch))
            #
            # with tf.train.MonitoredTrainingSession(
            # checkpoint_dir=FLAGS.train_dir,
            # hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
            #  tf.train.NanTensorHook(loss),
            #  _LoggerHook()],
            # config=tf.ConfigProto(
            # log_device_placement=FLAGS.log_device_placement)) as mon_sess:
            # while not mon_sess.should_stop():
            # print("Running session")
            # mon_sess.run(train_op)

def write_empty_inference_graph():
    with tf.Graph().as_default():                
        print("Preparing input")            
        images = tf.placeholder(tf.float32, [1, ocr_input.IMAGE_WIDTH, ocr_input.IMAGE_HEIGHT, ocr_input.IMAGE_DEPTH])
        logits, timesteps = ocr.inference(images, 1, train=True)
        decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, timesteps)
        log_prob = tf.identity(log_prob, name="decoded_log_prob")
        decoded = tf.cast(decoded[0], tf.int32, name="decoded_indexes")
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        sess = tf.Session()
        sess.run(init_op)
        tf.train.write_graph(sess.graph_def, FLAGS.train_dir, 'minimal_graph.proto', as_text=False)
        tf.train.write_graph(sess.graph_def, FLAGS.train_dir, 'minimal_graph.txt', as_text=True)
        

def main(argv=None):  # pylint: disable=unused-argument
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    write_empty_inference_graph()
    train()


if __name__ == '__main__':
    tf.app.run()
