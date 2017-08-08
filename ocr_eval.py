"""Evaluation for ocr."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time

import tensorflow as tf

import ocr
import os
import gc

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', 'eval_logs',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'test',
                           """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_string('best_dir', 'best_checkpoints',
                           """Directory where to write best snapshots.""")
tf.app.flags.DEFINE_string('checkpoint_dir', 'train_logs',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 1,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 6271,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', False,
                            """Whether to run eval only once.""")

tested_checkpoints = []
best_accuracy = 9999999


def eval_once(saver, summary_writer, ler, summary_op):
    """Run Eval once.
  
    Args:
      saver: Saver.
      summary_writer: Summary writer.
      ler: Top K op.
      summary_op: Summary op.
    """
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            # Assuming model_checkpoint_path looks something like:
            #   /my-favorite-path/ocr_train/model.ckpt-0,
            # extract global_step from it.
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            if global_step in tested_checkpoints:
                sess.close()
                return
            saver.restore(sess, ckpt.model_checkpoint_path)
            tested_checkpoints.append(global_step)
        else:
            print('No checkpoint file found')
            return

        # Start the queue runners.
        coord = tf.train.Coordinator()
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                                 start=True))

            num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.eval_batch_size))
            step = 0
            mean_ler = 0
            while step < num_iter and not coord.should_stop():
                ler_res = sess.run(ler)
                mean_ler += ler_res
                step += 1

            precision = mean_ler / step
            status_string = "{} Step: {} Val LER: {} "
            print(status_string.format(datetime.now(), global_step, precision))

            summary = tf.Summary()
            summary.ParseFromString(sess.run(summary_op))
            summary.value.add(tag='Val LER', simple_value=precision)
            summary_writer.add_summary(summary, global_step)

            global best_accuracy
            if precision < best_accuracy:
                best_accuracy = precision
                print("Saving new best checkpoint")
                saver.save(sess, os.path.join(FLAGS.best_dir, "checkpoint_ler_" + str(precision) + "_"))
        except Exception as e:  # pylint: disable=broad-except
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)
        sess.close()
    gc.collect()


def evaluate():
    """Eval ocr for a number of steps."""
    with tf.Graph().as_default() as g:
        images, labels, seq_lengths = ocr.inputs()
        logits, timesteps = ocr.inference(images, FLAGS.eval_batch_size, train=True)
        ler = ocr.create_label_error_rate(logits, labels, timesteps)
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        config = tf.ConfigProto(
            device_count={'GPU': 0}
        )
        sess = tf.Session(config=config)
        sess.run(init_op)

        saver = tf.train.Saver()

        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)

        while True:
            eval_once(saver, summary_writer, ler, summary_op)
            if FLAGS.run_once:
                break
            # print("Waiting for next evaluation for " + str(FLAGS.eval_interval_secs) + " sec")
            time.sleep(FLAGS.eval_interval_secs)


def main(argv=None):  # pylint: disable=unused-argument
    os.system('export CUDA_VISIBLE_DEVICES=""')
    if tf.gfile.Exists(FLAGS.eval_dir):
        tf.gfile.DeleteRecursively(FLAGS.eval_dir)
    tf.gfile.MakeDirs(FLAGS.eval_dir)
    evaluate()


if __name__ == '__main__':
    tf.app.run()
