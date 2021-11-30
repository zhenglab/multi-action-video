# Copyright 2017 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""A simple demonstration of running VGGish in inference mode.

This is intended as a toy example that demonstrates how the various building
blocks (feature extraction, model definition and loading, postprocessing) work
together in an inference context.

A WAV file (assumed to contain signed 16-bit PCM samples) is read in, converted
into log mel spectrogram examples, fed into VGGish, the raw embedding output is
whitened and quantized, and the postprocessed embeddings are optionally written
in a SequenceExample to a TFRecord file (using the same format as the embedding
features released in AudioSet).

Usage:
  # Run a WAV file through the model and print the embeddings. The model
  # checkpoint is loaded from vggish_model.ckpt and the PCA parameters are
  # loaded from vggish_pca_params.npz in the current directory.
  $ python vggish_inference_demo.py --wav_file /path/to/a/wav/file

  # Run a WAV file through the model and also write the embeddings to
  # a TFRecord file. The model checkpoint and PCA parameters are explicitly
  # passed in as well.
  $ python vggish_inference_demo.py --wav_file /path/to/a/wav/file \
                                    --tfrecord_file /path/to/tfrecord/file \
                                    --checkpoint /path/to/model/checkpoint \
                                    --pca_params /path/to/pca/params

  # Run a built-in input (a sine wav) through the model and print the
  # embeddings. Associated model files are read from the current directory.
  $ python vggish_inference_demo.py
"""

from __future__ import print_function

import numpy as np
import six
import soundfile
import tensorflow.compat.v1 as tf

import vggish_input
import vggish_params
import vggish_postprocess
import vggish_slim

import collections
import pickle
import os

flags = tf.app.flags

flags.DEFINE_string(
    'audio_dir', None,
    'Path to a audio dir. ')

flags.DEFINE_string(
    'label_file', None,
    'Path to a label file. ')

flags.DEFINE_string(
    'write_file', None,
    'Path to a write file. ')

flags.DEFINE_string(
    'checkpoint', 'vggish_model.ckpt',
    'Path to the VGGish checkpoint file.')

flags.DEFINE_string(
    'pca_params', 'vggish_pca_params.npz',
    'Path to the VGGish PCA parameters file.')

# flags.DEFINE_string(
#     'tfrecord_file', None,
#     'Path to a TFRecord file where embeddings will be written.')

FLAGS = flags.FLAGS


def main(_):
    audio_dir = FLAGS.audio_dir
    label_file = FLAGS.label_file
    write_file = FLAGS.write_file
    with open(label_file, "r") as audio_labels:
        audio_feature_dict = collections.OrderedDict()
        write_audio_count = 0
        
        for line in audio_labels:
            line = line.strip("\n").strip()
            video_path = line.split(',', 1)[0].strip()
            audio_path = os.path.splitext(video_path)[0] + '.wav'
            audio_full_path = os.path.join(audio_dir, audio_path)
            if not os.path.exists(audio_full_path):
                print("{} not found.".format(audio_full_path))
                continue

            examples_batch = vggish_input.wavfile_to_examples(audio_full_path)
#             print(examples_batch)
        
            # Prepare a postprocessor to munge the model embeddings.
            pproc = vggish_postprocess.Postprocessor(FLAGS.pca_params)
        
            # If needed, prepare a record writer to store the postprocessed embeddings.
#             writer = tf.python_io.TFRecordWriter(
#                 FLAGS.tfrecord_file) if FLAGS.tfrecord_file else None
        
            with tf.Graph().as_default(), tf.Session() as sess:
                # Define the model in inference mode, load the checkpoint, and
                # locate input and output tensors.
                vggish_slim.define_vggish_slim(training=False)
                vggish_slim.load_vggish_slim_checkpoint(sess, FLAGS.checkpoint)
                features_tensor = sess.graph.get_tensor_by_name(
                    vggish_params.INPUT_TENSOR_NAME)
                embedding_tensor = sess.graph.get_tensor_by_name(
                    vggish_params.OUTPUT_TENSOR_NAME)
            
                # Run inference and postprocessing.
                [embedding_batch] = sess.run([embedding_tensor],
                                             feed_dict={features_tensor: examples_batch})
#                 print(type(embedding_batch))
#                 print(embedding_batch.shape)
#                 print(embedding_batch)
                postprocessed_batch = pproc.postprocess(embedding_batch)
#                 print(type(postprocessed_batch))
#                 print(postprocessed_batch.shape)
#                 print(postprocessed_batch)
                if isinstance(postprocessed_batch, np.ndarray) and postprocessed_batch.shape == (3,128):
                    if (postprocessed_batch[0] == postprocessed_batch[1]).all() and \
                        (postprocessed_batch[0] == postprocessed_batch[2]).all():
                        print("{} is a slient audio (3 secs), skip ...".format(audio_path))
                        continue
                    if (postprocessed_batch[0] == postprocessed_batch[1]).all() or \
                        (postprocessed_batch[0] == postprocessed_batch[2]).all() or \
                         (postprocessed_batch[1] == postprocessed_batch[2]).all():
                        print("{} is a slient audio (2 secs), skip ...".format(audio_path))
                        continue

                    audio_feature_dict[audio_path] = postprocessed_batch.reshape(-1)
#                     print(type(audio_feature_dict[audio_path]))
#                     print(audio_feature_dict[audio_path].shape)
                    write_audio_count += 1
                    print("++++++++++++------------ processed audio number is {}".format(write_audio_count))
                    if write_audio_count % 3000 == 0:
                        with open("{}_{}".format(str(write_audio_count), write_file), 'wb') as f:
                            pickle.dump(audio_feature_dict, f, protocol=4)
        
        with open("{}_{}".format(str(write_audio_count), write_file), 'wb') as f:
            pickle.dump(audio_feature_dict, f, protocol=4)

if __name__ == '__main__':
    tf.app.run()
