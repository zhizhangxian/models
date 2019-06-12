
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cProfile
import functools
import pstats
import threading
import timeit
import uuid

from absl import app
from absl import flags
import tensorflow as tf

FLAGS = flags.FLAGS


def main(argv):
  output_dir = "/tmp/keras_imn_debug"
  if tf.io.gfile.exists(output_dir):
    tf.io.gfile.rmtree(output_dir)

  from official.bert.benchmark import bert_benchmark

  st = timeit.default_timer()
  test_class = bert_benchmark.BertClassifyBenchmarkReal(output_dir=output_dir)
  test_class.benchmark_2_gpu_mrpc()
  print("Overall time: {:.2f} sec".format(timeit.default_timer() - st))


if __name__ == '__main__':
  app.run(main)

