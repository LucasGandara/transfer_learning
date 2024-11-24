#!/usr/bin/env python3
# Authors: Lucas G. #

import tensorflow as tf


class MetricsWriter(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.summary_writer = tf.summary.create_file_writer(
            cfg["tensorboard_base_log_dir"]
        )
        self.step = 0

    def update_stats(self, **stats: dict):
        for key, value in stats.items():
            self.writer.add_scalar(key, value, self.step)
        self.writer.flush()

    def close_writer(self):
        self.summary_writer.close()
