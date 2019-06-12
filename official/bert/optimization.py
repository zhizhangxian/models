# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Functions and classes related to optimization (weight updates)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re

import tensorflow as tf


class WarmUp(tf.keras.optimizers.schedules.LearningRateSchedule):
  """Applys a warmup schedule on a given learning rate decay schedule."""

  def __init__(
      self,
      initial_learning_rate,
      decay_schedule_fn,
      warmup_steps,
      power=1.0,
      name=None):
    super(WarmUp, self).__init__()
    self.initial_learning_rate = initial_learning_rate
    self.warmup_steps = warmup_steps
    self.power = power
    self.decay_schedule_fn = decay_schedule_fn
    self.name = name

  def __call__(self, step):
    with tf.name_scope(self.name or 'WarmUp') as name:
      # Implements polynomial warmup. i.e., if global_step < warmup_steps, the
      # learning rate will be `global_step/num_warmup_steps * init_lr`.
      global_step_float = tf.cast(step, tf.float32)
      warmup_steps_float = tf.cast(self.warmup_steps, tf.float32)
      warmup_percent_done = global_step_float / warmup_steps_float
      warmup_learning_rate = (
          self.initial_learning_rate *
          tf.math.pow(warmup_percent_done, self.power))
      return tf.cond(global_step_float < warmup_steps_float,
                     lambda: warmup_learning_rate,
                     lambda: self.decay_schedule_fn(step),
                     name=name)

  def get_config(self):
    return {
        'initial_learning_rate': self.initial_learning_rate,
        'decay_schedule_fn': self.decay_schedule_fn,
        'warmup_steps': self.warmup_steps,
        'power': self.power,
        'name': self.name
    }


def create_optimizer(init_lr, num_train_steps, num_warmup_steps):
  """Creates an optimizer with learning rate schedule."""
  # Implements linear decay of the learning rate.
  learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
      initial_learning_rate=init_lr,
      decay_steps=num_train_steps,
      end_learning_rate=0.0)
  if num_warmup_steps:
    learning_rate_fn = WarmUp(initial_learning_rate=init_lr,
                              decay_schedule_fn=learning_rate_fn,
                              warmup_steps=num_warmup_steps)
  optimizer = AdamWeightDecay(
      learning_rate=learning_rate_fn,
      weight_decay_rate=0.01,
      beta_1=0.9,
      beta_2=0.999,
      epsilon=1e-6,
      exclude_from_weight_decay=['layer_norm', 'bias'])
  return optimizer


class AdamWithReuse(tf.keras.optimizers.Adam):
  """Manaully cache constants.

  Adjust Adam implementation until a general fix is in.
  """

  def __init__(self, *args, **kwargs):
    super(AdamWithReuse, self).__init__(*args, **kwargs)
    self._apply_cache = ApplyCache(self._create_step_constants)

  def apply_gradients(self, grads_and_vars, name=None):
    with self._apply_cache:
      return super(AdamWithReuse, self).apply_gradients(grads_and_vars, name)

  def _create_step_constants(self, device, var_dtype):
    _ = device  # device is only passed as a cache key.

    lr_t = self._decayed_lr(var_dtype)
    local_step = math_ops.cast(self.iterations + 1, var_dtype)

    beta_1_t = self._get_hyper('beta_1', var_dtype)
    beta_2_t = self._get_hyper('beta_2', var_dtype)
    beta_1_power = math_ops.pow(beta_1_t, local_step)
    beta_2_power = math_ops.pow(beta_2_t, local_step)

    return dict(
        lr_t=lr_t,
        lr=lr_t * math_ops.sqrt(1 - beta_2_power) / (1 - beta_1_power),
        epsilon_t=ops.convert_to_tensor(self.epsilon, var_dtype),
        beta_1_t=beta_1_t,
        beta_1_power=beta_1_power,
        one_minus_beta_1_t=1 - beta_1_t,
        beta_2_t=beta_2_t,
        beta_2_power=beta_2_power,
        one_minus_beta_2_t=1 - beta_2_t)

  def _resource_apply_dense(self, grad, var):
    m = self.get_slot(var, 'm')
    v = self.get_slot(var, 'v')

    constants = self._apply_cache[var.device, var.dtype.base_dtype]

    if not self.amsgrad:
      return training_ops.resource_apply_adam(
          var.handle,
          m.handle,
          v.handle,
          constants.beta_1_power,
          constants.beta_2_power,
          constants.lr_t,
          constants.beta_1_t,
          constants.beta_2_t,
          constants.epsilon_t,
          grad,
          use_locking=self._use_locking)
    else:
      vhat = self.get_slot(var, 'vhat')
      return training_ops.resource_apply_adam_with_amsgrad(
          var.handle,
          m.handle,
          v.handle,
          vhat.handle,
          constants.beta_1_power,
          constants.beta_2_power,
          constants.lr_t,
          constants.beta_1_t,
          constants.beta_2_t,
          constants.epsilon_t,
          grad,
          use_locking=self._use_locking)

  def _resource_apply_sparse(self, grad, var, indices):
    constants = self._apply_cache[var.device, var.dtype.base_dtype]

    # m_t = beta1 * m + (1 - beta1) * g_t
    m = self.get_slot(var, 'm')
    m_scaled_g_values = grad * constants.one_minus_beta_1_t
    m_t = state_ops.assign(m, m * constants.beta_1_t,
                           use_locking=self._use_locking)
    with ops.control_dependencies([m_t]):
      m_t = self._resource_scatter_add(m, indices, m_scaled_g_values)

    # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
    v = self.get_slot(var, 'v')
    v_scaled_g_values = (grad * grad) * constants.one_minus_beta_2_t
    v_t = state_ops.assign(v, v * constants.beta_2_t,
                           use_locking=self._use_locking)
    with ops.control_dependencies([v_t]):
      v_t = self._resource_scatter_add(v, indices, v_scaled_g_values)

    if not self.amsgrad:
      v_sqrt = math_ops.sqrt(v_t)
      var_update = state_ops.assign_sub(
          var, constants.lr * m_t / (v_sqrt + constants.epsilon_t),
          use_locking=self._use_locking)
      return control_flow_ops.group(*[var_update, m_t, v_t])
    else:
      v_hat = self.get_slot(var, 'vhat')
      v_hat_t = math_ops.maximum(v_hat, v_t)
      with ops.control_dependencies([v_hat_t]):
        v_hat_t = state_ops.assign(
            v_hat, v_hat_t, use_locking=self._use_locking)
      v_hat_sqrt = math_ops.sqrt(v_hat_t)
      var_update = state_ops.assign_sub(
          var,
          constants.lr * m_t / (v_hat_sqrt + constants.epsilon_t),
          use_locking=self._use_locking)
      return control_flow_ops.group(*[var_update, m_t, v_t, v_hat_t])


class AdamWeightDecay(AdamWithReuse):
  """Adam enables L2 weight decay and clip_by_global_norm on gradients.

  Just adding the square of the weights to the loss function is *not* the
  correct way of using L2 regularization/weight decay with Adam, since that will
  interact with the m and v parameters in strange ways.

  Instead we want ot decay the weights in a manner that doesn't interact with
  the m/v parameters. This is equivalent to adding the square of the weights to
  the loss with plain (non-momentum) SGD.
  """

  def __init__(self,
               learning_rate=0.001,
               beta_1=0.9,
               beta_2=0.999,
               epsilon=1e-7,
               amsgrad=False,
               weight_decay_rate=0.0,
               exclude_from_weight_decay=None,
               name='AdamWeightDecay',
               **kwargs):
    super(AdamWeightDecay, self).__init__(
        learning_rate, beta_1, beta_2, epsilon, amsgrad, name, **kwargs)
    self._set_hyper('weight_decay_rate', weight_decay_rate)
    self._exclude_from_weight_decay = exclude_from_weight_decay

  @classmethod
  def from_config(cls, config):
    """Creates an optimizer from its config with WarmUp custom object."""
    custom_objects = {'WarmUp': WarmUp}
    return super(AdamWeightDecay, cls).from_config(
        config, custom_objects=custom_objects)

  def _decay_weights_op(self, var, learning_rate):
    do_decay = self._do_use_weight_decay(var.name)
    if do_decay:
      return var.assign_sub(
          learning_rate * var *
          self._get_hyper('weight_decay_rate'),
          use_locking=self._use_locking)
    return tf.no_op()

  def apply_gradients(self, grads_and_vars, name=None):
    grads, tvars = list(zip(*grads_and_vars))
    (grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)
    return super(AdamWeightDecay, self).apply_gradients(zip(grads, tvars))

  def _resource_apply_dense(self, grad, var):
    lr_t = self._apply_cache[var.device, var.dtype.base_dtype].lr_t
    with tf.control_dependencies([self._decay_weights_op(var, lr_t)]):
      return super(AdamWeightDecay, self)._resource_apply_dense(
          grad, var)

  def _resource_apply_sparse(self, grad, var, indices):
    lr_t = self._apply_cache[var.device, var.dtype.base_dtype].lr_t
    with tf.control_dependencies([self._decay_weights_op(var, lr_t)]):
      return super(AdamWeightDecay, self)._resource_apply_sparse(
          grad, var, indices)

  def get_config(self):
    config = super(AdamWeightDecay, self).get_config()
    config.update({
        'weight_decay_rate':
            self._serialize_hyperparameter('weight_decay_rate'),
    })
    return config

  def _do_use_weight_decay(self, param_name):
    """Whether to use L2 weight decay for `param_name`."""
    if self.weight_decay_rate == 0:
      return False
    if self._exclude_from_weight_decay:
      for r in self._exclude_from_weight_decay:
        if re.search(r, param_name) is not None:
          return False
    return True


class SubCache(object):

  def __init__(self, init_fn, keys):
    self.values = init_fn(*keys)

  def __getattr__(self, name):
    return self.values[name]


class ApplyCache(object):

  def __init__(self, init_fn):
    self._init_fn = init_fn
    self._sub_caches = None

  def __enter__(self):
    self._sub_caches = {}

  def __exit__(self, exc_type, exc_val, exc_tb):
    self._sub_caches = None

  def __getitem__(self, keys):
    if self._sub_caches is None:
      raise ValueError('Cache is not valid outside of gradient apply.')

    sub_cache = self._sub_caches.get(keys)
    if sub_cache is None:
      self._sub_caches[keys] = sub_cache = SubCache(self._init_fn, keys)
    return sub_cache

