# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

"""带有注意力机制的Sequence-to-sequence 模型."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
datautil = __import__("9-33  datautil")
import datautil as data_utils


class Seq2SeqModel(object):
  """带有注意力机制并且具有multiple buckets的Sequence-to-sequence 模型.
    这个类实现了一个多层循环网络组成的编码器和一个具有注意力机制的解码器.完全是按照论文：
    http://arxiv.org/abs/1412.7449 - 中所描述的机制实现。更多细节信息可以参看论文内容
    这个class 除了使用LSTM cells还可以使用GRU cells, 还使用了sampled softmax 来
    处理大词汇量的输出. 在论文http://arxiv.org/abs/1412.2007中的第三节描述了
    sampled softmax。在论文http://arxiv.org/abs/1409.0473里面还有一个关于这个模
    型的一个单层的使用双向RNN编码器的版本。
  """

  def __init__(self,
               source_vocab_size,
               target_vocab_size,
               buckets,
               size,
               num_layers,
               dropout_keep_prob,
               max_gradient_norm,
               batch_size,
               learning_rate,
               learning_rate_decay_factor,
               use_lstm=False,
               num_samples=512,
               forward_only=False,
               dtype=tf.float32):
    """创建模型.

            Args:
            source_vocab_size:原词汇的大小.
            target_vocab_size:目标词汇的大小.
            buckets: 一个 (I, O)的list, I 代表输入的最大长度，O代表输出的最大长度，例如[(2, 4), (8, 16)].
            size: 模型中每层的units个数.
            num_layers: 模型的层数.
            max_gradient_norm: 截断梯度的阀值.
            batch_size: 训练中的批次数据大小;
            learning_rate: 开始学习率.
            learning_rate_decay_factor: 退化学习率的衰减参数.
            use_lstm: 如果true, 使用 LSTM cells 替代GRU cells.
            num_samples: sampled softmax的样本个数.
            forward_only: 如果设置了, 模型只有正向传播.
            dtype: internal variables的类型.
    """

    self.source_vocab_size = source_vocab_size
    self.target_vocab_size = target_vocab_size
    self.buckets = buckets
    self.batch_size = batch_size
    self.dropout_keep_prob_output = dropout_keep_prob
    self.dropout_keep_prob_input = dropout_keep_prob
    self.learning_rate = tf.Variable(
        float(learning_rate), trainable=False, dtype=dtype)
    self.learning_rate_decay_op = self.learning_rate.assign(
        self.learning_rate * learning_rate_decay_factor)
    self.global_step = tf.Variable(0, trainable=False)

    # 如果使用 sampled softmax, 需要一个输出的映射.
    output_projection = None
    softmax_loss_function = None
    # 当采样数小于vocabulary size 是Sampled softmax 才有意义.
    if num_samples > 0 and num_samples < self.target_vocab_size:
      w_t = tf.get_variable("proj_w", [self.target_vocab_size, size], dtype=dtype)
      w = tf.transpose(w_t)
      b = tf.get_variable("proj_b", [self.target_vocab_size], dtype=dtype)
      output_projection = (w, b)

      def sampled_loss(labels, logits):
        labels = tf.reshape(labels, [-1, 1])
         #需要使用 32bit的浮点数类型来计算sampled_softmax_loss,才会避免数值的不稳定性发生.
        local_w_t = tf.cast(w_t, tf.float32)
        local_b = tf.cast(b, tf.float32)
        local_inputs = tf.cast(logits, tf.float32)
        return tf.cast(
            tf.nn.sampled_softmax_loss(
                weights=local_w_t,
                biases=local_b,
                labels=labels,
                inputs=local_inputs,
                num_sampled=num_samples,
                num_classes=self.target_vocab_size),
            dtype)
      softmax_loss_function = sampled_loss


    # The seq2seq function: we use embedding for the input and attention.
    def seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
#      cells = []
#      for i in range(num_layers):
#            with tf.variable_scope('RNN_{}'.format(i)):
#                cells.append(tf.contrib.rnn.GRUCell(size))
#      cell = tf.contrib.rnn.MultiRNNCell(cells)
      
      with tf.variable_scope("GRU") as scope:
          cell = tf.contrib.rnn.DropoutWrapper(
              tf.contrib.rnn.GRUCell(size),
                input_keep_prob=self.dropout_keep_prob_input,
                output_keep_prob=self.dropout_keep_prob_output)
          if num_layers > 1:
              cell = tf.contrib.rnn.MultiRNNCell([cell] * num_layers)       
      
      
      print("new a cell")
      return tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
          encoder_inputs,
          decoder_inputs,
          cell,
          num_encoder_symbols=source_vocab_size,
          num_decoder_symbols=target_vocab_size,
          embedding_size=size,
          output_projection=output_projection,
          feed_previous=do_decode,
          dtype=dtype)

    #  注入数据.
    self.encoder_inputs = []
    self.decoder_inputs = []
    self.target_weights = []
    for i in xrange(buckets[-1][0]):  #最后的bucket 是最大的.
      self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                name="encoder{0}".format(i)))
    for i in xrange(buckets[-1][1] + 1):
      self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                name="decoder{0}".format(i)))
      self.target_weights.append(tf.placeholder(dtype, shape=[None],
                                                name="weight{0}".format(i)))

    #将解码器移动一位得到targets.
    targets = [self.decoder_inputs[i + 1]
               for i in xrange(len(self.decoder_inputs) - 1)]

    # 训练的输出和loss定义.
    if forward_only:
      self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(
          self.encoder_inputs, self.decoder_inputs, targets,
          self.target_weights, buckets, lambda x, y: seq2seq_f(x, y, True),
          softmax_loss_function=softmax_loss_function)
      # 如果使用了输出映射, 需要为解码器映射输出处理.
      if output_projection is not None:
        for b in xrange(len(buckets)):
          self.outputs[b] = [
              tf.matmul(output, output_projection[0]) + output_projection[1]
              for output in self.outputs[b]
          ]
    else:
      self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(
          self.encoder_inputs, self.decoder_inputs, targets,
          self.target_weights, buckets,
          lambda x, y: seq2seq_f(x, y, False),
          softmax_loss_function=softmax_loss_function)

    # 梯度下降更新操作.
    params = tf.trainable_variables()
    if not forward_only:
      self.gradient_norms = []
      self.updates = []
      opt = tf.train.GradientDescentOptimizer(self.learning_rate)
      for b in xrange(len(buckets)):
        gradients = tf.gradients(self.losses[b], params)
        clipped_gradients, norm = tf.clip_by_global_norm(gradients,
                                                         max_gradient_norm)
        self.gradient_norms.append(norm)
        self.updates.append(opt.apply_gradients(
            zip(clipped_gradients, params), global_step=self.global_step))

    self.saver = tf.train.Saver(tf.global_variables())

  def step(self, session, encoder_inputs, decoder_inputs, target_weights,
           bucket_id, forward_only):
    """注入给定输入数据步骤。	
      Args:
          session: tensorflow 所使用的session.
          encoder_inputs: 用来注入encoder 输入数据的numpy int vectors类型的list。
          decoder_inputs: 用来注入decoder输入数据的numpy int vectors类型的list。
          target_weights: 用来注入target weights的numpy float vectors类型的list.
          bucket_id: which bucket of the model to use.
          forward_only: 只进行正向传播.
          
	    Returns:
             一个由gradient norm (不做反向时为none),average perplexity, and the outputs组成的triple.
         
	    Raises:
             ValueError:如果 encoder_inputs, decoder_inputs, 或者是target_weights 的长度与指定bucket_id 的bucket size不符合.
    """

    # 检查长度..
    encoder_size, decoder_size = self.buckets[bucket_id]
    if len(encoder_inputs) != encoder_size:
      raise ValueError("Encoder length must be equal to the one in bucket,"
                       " %d != %d." % (len(encoder_inputs), encoder_size))
    if len(decoder_inputs) != decoder_size:
      raise ValueError("Decoder length must be equal to the one in bucket,"
                       " %d != %d." % (len(decoder_inputs), decoder_size))
    if len(target_weights) != decoder_size:
      raise ValueError("Weights length must be equal to the one in bucket,"
                       " %d != %d." % (len(target_weights), decoder_size))

    # 定义Input feed
    input_feed = {}
    for l in xrange(encoder_size):
      input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
    for l in xrange(decoder_size):
      input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
      input_feed[self.target_weights[l].name] = target_weights[l]

    # Since our targets are decoder inputs shifted by one, we need one more.
    last_target = self.decoder_inputs[decoder_size].name
    input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)

    # 定义Output feed
    if not forward_only:
      output_feed = [self.updates[bucket_id],  # Update Op that does SGD.
                     self.gradient_norms[bucket_id],  # Gradient norm.
                     self.losses[bucket_id]]  # Loss for this batch.
    else:
      output_feed = [self.losses[bucket_id]]  # Loss for this batch.
      for l in xrange(decoder_size):  # Output logits.
        output_feed.append(self.outputs[bucket_id][l])

    outputs = session.run(output_feed, input_feed)
    if not forward_only:
      return outputs[1], outputs[2], None  # Gradient norm, loss, no outputs.
    else:
      return None, outputs[0], outputs[1:]  # No gradient norm, loss, outputs.   

  def get_batch(self, data, bucket_id):
    """在迭代训练过程中，从指定 bucket中获得一个随机批次数据.
	    Args:
	      data: 一个大小为len(self.buckets)的tuple，包含了创建一个batch中的输入输出的
	        lists.
	      bucket_id: 整型, 指定从哪个bucket中取数据.
	    Returns:
	      方便以后调用的 triple (encoder_inputs, decoder_inputs, target_weights) 
    """
    encoder_size, decoder_size = self.buckets[bucket_id]
    encoder_inputs, decoder_inputs = [], []

    # 获得一个随机批次的数据作为编码器与解码器的输入,
    # 如果需要时会有pad操作, 同时反转encoder的输入顺序，并且为decoder添加GO.
    for _ in xrange(self.batch_size):
      encoder_input, decoder_input = random.choice(data[bucket_id])

      # pad和反转Encoder 的输入数据..
      encoder_pad = [data_utils.PAD_ID] * (encoder_size - len(encoder_input))
      encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))

      # 为Decoder输入数据添加一个额外的"GO", 并且进行pad.
      decoder_pad_size = decoder_size - len(decoder_input) - 1
      decoder_inputs.append([data_utils.GO_ID] + decoder_input +
                            [data_utils.PAD_ID] * decoder_pad_size)

    # 从上面选择好的数据中创建 batch-major vectors.
    batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []

    # Batch encoder inputs are just re-indexed encoder_inputs.
    for length_idx in xrange(encoder_size):
      batch_encoder_inputs.append(
          np.array([encoder_inputs[batch_idx][length_idx]
                    for batch_idx in xrange(self.batch_size)], dtype=np.int32))

    # Batch decoder inputs are re-indexed decoder_inputs, we create weights.
    for length_idx in xrange(decoder_size):
      batch_decoder_inputs.append(
          np.array([decoder_inputs[batch_idx][length_idx]
                    for batch_idx in xrange(self.batch_size)], dtype=np.int32))

      # 定义target_weights 变量，默认是1，如果对应的targets是padding，就为0.
      batch_weight = np.ones(self.batch_size, dtype=np.float32)
      for batch_idx in xrange(self.batch_size):
          # 如果对应的输出target 是一个 PAD符号，就将weight设为0.
          # 将decoder_input向前移动1位得到对应的target.
        if length_idx < decoder_size - 1:
          target = decoder_inputs[batch_idx][length_idx + 1]
        if length_idx == decoder_size - 1 or target == data_utils.PAD_ID:
          batch_weight[batch_idx] = 0.0
      batch_weights.append(batch_weight)
    return batch_encoder_inputs, batch_decoder_inputs, batch_weights
