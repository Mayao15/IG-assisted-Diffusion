# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""All functions related to loss computation and optimization.
"""

import flax
import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
from models import utils as mutils
from sde_lib import VESDE, VPSDE
from utils import batch_mul


def get_optimizer(config):
  """Returns a flax optimizer object based on `config`."""
  if config.optim.optimizer == 'Adam':
    optimizer = flax.optim.Adam(beta1=config.optim.beta1, eps=config.optim.eps,
                                weight_decay=config.optim.weight_decay)
  else:
    raise NotImplementedError(
      f'Optimizer {config.optim.optimizer} not supported yet!')

  return optimizer


def optimization_manager(config):
  """Returns an optimize_fn based on `config`."""

  def optimize_fn(state,
                  grad,
                  warmup=config.optim.warmup,
                  grad_clip=config.optim.grad_clip):
    """Optimizes with warmup and gradient clipping (disabled if negative)."""
    lr = state.lr
    if warmup > 0:
      lr = lr * jnp.minimum(state.step / warmup, 1.0)
    if grad_clip >= 0:
      # Compute global gradient norm
      grad_norm = jnp.sqrt(
        sum([jnp.sum(jnp.square(x)) for x in jax.tree_leaves(grad)]))
      # Clip gradient
      clipped_grad = jax.tree_map(
        lambda x: x * grad_clip / jnp.maximum(grad_norm, grad_clip), grad)
    else:  # disabling gradient clipping if grad_clip < 0
      clipped_grad = grad
    return state.optimizer.apply_gradient(clipped_grad, learning_rate=lr)

  return optimize_fn


def get_sde_loss_fn(sde, model, train, reduce_mean=True, continuous=True, likelihood_weighting=True, eps=1e-5):
  """Create a loss function for training with arbirary SDEs.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    model: A `flax.linen.Module` object that represents the architecture of the score-based model.
    train: `True` for training loss and `False` for evaluation loss.
    reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
    continuous: `True` indicates that the model is defined to take continuous time steps. Otherwise it requires
      ad-hoc interpolation to take continuous time steps.
    likelihood_weighting: If `True`, weight the mixture of score matching losses
      according to https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended in our paper.
    eps: A `float` number. The smallest time step to sample from.

  Returns:
    A loss function.
  """
  reduce_op = jnp.mean if reduce_mean else lambda *args, **kwargs: 0.5 * jnp.sum(*args, **kwargs)

  def loss_fn(rng, params, states, batch, step):
    """Compute the loss function.

    Args:
      rng: A JAX random state.
      params: A dictionary that contains trainable parameters of the score-based model.
      states: A dictionary that contains mutable states of the score-based model.
      batch: A mini-batch of training data.

    Returns:
      loss: A scalar that represents the average loss value across the mini-batch.
      new_model_state: A dictionary that contains the mutated states of the score-based model.
    """

    # step = list(states.keys())
    # print(step)

    # score_fn = mutils.get_score_fn(sde, model, params, states, train=train, continuous=continuous, return_state=True)
    score_fn_no_state = mutils.get_score_fn(sde, model, params, states, train=train, continuous=continuous, return_state=False)
    data = batch['image']

    def score_jvp_fn(x, t, eps, step_rng, score_fn):
        fn = lambda inputs: score_fn(inputs, t, step_rng)
        score, score_vjp = jax.jvp(fn, (x, ), (eps, ))
        return score_vjp, score

    def inner_prod(x, y):
        x = x.reshape((x.shape[0], -1))
        y = y.reshape((y.shape[0], -1))
        return jnp.sum(x * y, axis=-1, keepdims=True)

    def grad_div_fn(x, t, eps, step_rng, score_fn):

        def score_jvp_fn(data):
            fn = lambda inputs: score_fn(inputs, t, rng=step_rng)
            score, score_jvp = jax.jvp(fn, (data, ), (eps, ))
            return jnp.sum(eps * score_jvp), (score_jvp, score)

        grad_div, (score_jvp, score) = jax.grad(score_jvp_fn, has_aux=True)(x)
        return grad_div, score_jvp, score

    def score_jvp_fn(x, t, eps, step_rng, score_fn):
        fn = lambda inputs: score_fn(inputs, t, step_rng)
        score, score_vjp = jax.jvp(fn, (x, ), (eps, ))
        return score_vjp, score

    def score_jvp_fn_t(x, t, eps, step_rng, score_fn):
        fn = lambda inputs: score_fn(x, inputs, step_rng)
        score, score_vjp = jax.jvp(fn, (t, ), (eps, ))
        return score_vjp, score

    def vgrad(f, x):
        y, vjp_fn = jax.vjp(f, x)
        return vjp_fn(jnp.ones(y.shape))[0]

    dim = int(np.prod(data.shape[1:]))

    rng, step_rng = random.split(rng)
    t = random.uniform(step_rng, (data.shape[0],), minval=eps, maxval=sde.T)

    rng, step_rng = random.split(rng)
    z = random.normal(step_rng, data.shape)

    # calculate mean and std with t + delta_t
    mean, std = sde.marginal_prob(data, t)
    perturbed_data = mean + batch_mul(std, z)

    rng, step_rng = random.split(rng)
    v = random.randint(step_rng, data.shape, minval=0, maxval=2).astype(jnp.float32) * 2 - 1

    # code borrowed from Lu et al.
    new_model_state = states
    score_grad_div, score_jvp, score = grad_div_fn(perturbed_data, t, v, step_rng,
                                                   score_fn_no_state)
    score_fixed = jax.lax.stop_gradient(score)
    std_2 = jnp.square(std).reshape((data.shape[0], 1))
    std = std.reshape((data.shape[0], 1))
    v_2 = inner_prod(v, v)

    cond_score = (z.reshape((data.shape[0], dim)) + score_fixed.reshape(
        (data.shape[0], dim)) * std).reshape((data.shape[0], dim))
    cond_score_v_prod_norm_square = jnp.square(inner_prod(cond_score, v))
    # Frobenius:
    losses_2_frob = ((score_jvp.reshape((data.shape[0], dim)) * std_2) + v.reshape(
        (data.shape[0], dim)) - (inner_prod(cond_score, v)) * cond_score)
    losses_2_frob = jnp.square(losses_2_frob) / float(dim)
    losses_2_frob = reduce_op(losses_2_frob.reshape((losses_2_frob.shape[0], -1)) , axis=-1)

    # Trace:
    score_div = inner_prod(score_jvp, v)
    losses_2 = score_div * std_2 + v_2 - cond_score_v_prod_norm_square
    losses_2 = losses_2 / float(dim)
    losses_2 = jnp.square(losses_2)
    losses_2 = reduce_op(losses_2.reshape((losses_2.shape[0], -1)) , axis=-1)

    losses_2 = losses_2 + losses_2_frob

    score_div_fixed = jax.lax.stop_gradient(score_div)
    score_jvp_fixed = jax.lax.stop_gradient(score_jvp)
    losses_3 = (std_2 * std * score_grad_div.reshape(
        (data.shape[0], dim)) + cond_score_v_prod_norm_square * cond_score -
                (std_2 * score_div_fixed + v_2) * cond_score - 2 * inner_prod(v, cond_score) *
                (std_2 * score_jvp_fixed.reshape((data.shape[0], dim)) + v.reshape(
                    (data.shape[0], dim))))
    losses_3 = losses_3 / float(dim)
    losses_3 = jnp.square(losses_3)
    losses_3 = reduce_op(losses_3.reshape((losses_3.shape[0], dim)) , axis=-1)
    # end of code borrowed from Lu et al.

    # 计算一些系数及他们的梯度
    alpha_grad = jax.vmap(jax.grad(lambda x, t: sde.mar(x, t)[0], 1))
    beta_grad = jax.vmap(jax.grad(lambda x, t: sde.mar(x, t)[1], 1))
    alpha, beta, _ = sde.mar(perturbed_data, t)
    drift, diffusion = sde.sde(perturbed_data, t)

    # 用于时间的梯度的采样向量。这实际是一个标量，考虑了batch_size之后才是一个向量。
    rng, step_rng = random.split(rng)
    vt = random.randint(step_rng, t.shape, minval=0, maxval=2).astype(jnp.float32) * 2 - 1

    # 分数函数计算的是epsilon_theta/beta_t
    # score_vjp, score = score_jvp_fn(perturbed_data, t, v, step_rng, score_fn_no_state)
    score_vjp_t, _ = score_jvp_fn_t(perturbed_data, t, vt, step_rng, score_fn_no_state)
    score_grad_div_fixed = jax.lax.stop_gradient(score_grad_div)
    temp1 = inner_prod(batch_mul(sde.d_drift(perturbed_data, t), score_fixed), v) # 这里应该是drift对x的梯度，是一个矩阵
    temp2 = inner_prod(drift, score_jvp_fixed)
    temp3 = batch_mul(diffusion ** 2 / 2, inner_prod(score_grad_div_fixed, v))
    temp4 = batch_mul(diffusion ** 2, inner_prod(score_fixed, score_jvp_fixed))

    loss_t = batch_mul(inner_prod(score_vjp_t, v) + batch_mul(
        - temp1 - temp2 + temp3 + temp4, vt), std)
    loss_t = jnp.square(loss_t / dim)
    loss_t = loss_t / jnp.max(loss_t)
    loss_t = reduce_op(loss_t.reshape((loss_t.shape[0], -1)), axis=-1)

    if not likelihood_weighting:
      losses = jnp.square(batch_mul(score, std) + z)
      losses = reduce_op(losses.reshape((losses.shape[0], -1)), axis=-1)
    else:
      g2 = sde.sde(jnp.zeros_like(data), t)[1] ** 2
      losses = jnp.square(score[0] + batch_mul(z, 1. / std))
      losses = reduce_op(losses.reshape((losses.shape[0], -1)), axis=-1) * g2

    losses = jnp.mean(losses)
    losses_2 = jnp.mean(losses_2)
    losses_3 = jnp.mean(losses_3)
    loss_t = jnp.mean(loss_t)
    # loss = jax.lax.cond(jnp.isnan(loss_t), lambda: losses + losses_2 + losses_3, lambda: losses + losses_2 + losses_3 + loss_t)
    loss = jax.lax.cond(
        train,
        lambda: jax.lax.cond(step > 10000, lambda: losses,
                      lambda: losses + losses_2 + losses_3 + loss_t * jnp.exp(-step / 50000)),
        lambda: jax.lax.cond(step > 10000, lambda: losses,
                      lambda: losses + losses_2 + losses_3 + loss_t))
        # if loss_t > losses:
    #   loss = losses + losses_2 + losses_3
    # else:
    #   loss = losses + losses_2 + losses_3 + loss_t
    return loss, (new_model_state, losses, losses_2, losses_3, loss_t)

  return loss_fn


def get_smld_loss_fn(vesde, model, train, reduce_mean=False):
  """Legacy code to reproduce previous results on SMLD(NCSN). Not recommended for new work."""
  assert isinstance(vesde, VESDE), "SMLD training only works for VESDEs."

  # Previous SMLD models assume descending sigmas
  smld_sigma_array = vesde.discrete_sigmas[::-1]
  reduce_op = jnp.mean if reduce_mean else lambda *args, **kwargs: 0.5 * jnp.sum(*args, **kwargs)

  def loss_fn(rng, params, states, batch):
    model_fn = mutils.get_model_fn(model, params, states, train=train)
    data = batch['image']
    rng, step_rng = random.split(rng)
    labels = random.choice(step_rng, vesde.N, shape=(data.shape[0],))
    sigmas = smld_sigma_array[labels]
    rng, step_rng = random.split(rng)
    noise = batch_mul(random.normal(step_rng, data.shape), sigmas)
    perturbed_data = noise + data
    rng, step_rng = random.split(rng)
    score, new_model_state = model_fn(perturbed_data, labels, rng=step_rng)
    target = -batch_mul(noise, 1. / (sigmas ** 2))
    losses = jnp.square(score - target)
    losses = reduce_op(losses.reshape((losses.shape[0], -1)), axis=-1) * sigmas ** 2
    loss = jnp.mean(losses)
    return loss, new_model_state

  return loss_fn


def get_ddpm_loss_fn(vpsde, model, train, reduce_mean=True):
  """Legacy code to reproduce previous results on DDPM. Not recommended for new work."""
  assert isinstance(vpsde, VPSDE), "DDPM training only works for VPSDEs."

  reduce_op = jnp.mean if reduce_mean else lambda *args, **kwargs: 0.5 * jnp.sum(*args, **kwargs)

  def loss_fn(rng, params, states, batch):
    model_fn = mutils.get_model_fn(model, params, states, train=train)
    data = batch['image']
    rng, step_rng = random.split(rng)
    labels = random.choice(step_rng, vpsde.N, shape=(data.shape[0],))
    sqrt_alphas_cumprod = vpsde.sqrt_alphas_cumprod
    sqrt_1m_alphas_cumprod = vpsde.sqrt_1m_alphas_cumprod
    rng, step_rng = random.split(rng)
    noise = random.normal(step_rng, data.shape)
    perturbed_data = batch_mul(sqrt_alphas_cumprod[labels], data) + \
                     batch_mul(sqrt_1m_alphas_cumprod[labels], noise)
    rng, step_rng = random.split(rng)
    score, new_model_state = model_fn(perturbed_data, labels, rng=step_rng)
    losses = jnp.square(score - noise)
    losses = reduce_op(losses.reshape((losses.shape[0], -1)), axis=-1)
    loss = jnp.mean(losses)
    return loss, new_model_state

  return loss_fn


def get_step_fn(sde, model, train, optimize_fn=None, reduce_mean=False, continuous=True, likelihood_weighting=False):
  """Create a one-step training/evaluation function.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    model: A `flax.linen.Module` object that represents the architecture of the score-based model.
    train: `True` for training and `False` for evaluation.
    optimize_fn: An optimization function.
    reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
    continuous: `True` indicates that the model is defined to take continuous time steps.
    likelihood_weighting: If `True`, weight the mixture of score matching losses according to
      https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended by our paper.

  Returns:
    A one-step function for training or evaluation.
  """
  if continuous:
    loss_fn = get_sde_loss_fn(sde, model, train, reduce_mean=reduce_mean,
                              continuous=True, likelihood_weighting=likelihood_weighting)
  else:
    assert not likelihood_weighting, "Likelihood weighting is not supported for original SMLD/DDPM training."
    if isinstance(sde, VESDE):
      loss_fn = get_smld_loss_fn(sde, model, train, reduce_mean=reduce_mean)
    elif isinstance(sde, VPSDE):
      loss_fn = get_ddpm_loss_fn(sde, model, train, reduce_mean=reduce_mean)
    else:
      raise ValueError(f"Discrete training for {sde.__class__.__name__} is not recommended.")

  def step_fn(carry_state, batch):
    """Running one step of training or evaluation.

    This function will undergo `jax.lax.scan` so that multiple steps can be pmapped and jit-compiled together
    for faster execution.

    Args:
      carry_state: A tuple (JAX random state, `flax.struct.dataclass` containing the training state).
      batch: A mini-batch of training/evaluation data.

    Returns:
      new_carry_state: The updated tuple of `carry_state`.
      loss: The average loss value of this state.
    """

    (rng, state) = carry_state
    rng, step_rng = jax.random.split(rng)
    grad_fn = jax.value_and_grad(loss_fn, argnums=1, has_aux=True)
    if train:
      params = state.optimizer.target
      states = state.model_state
      (loss, (new_model_state, losses_1, losses_2, losses_3, loss_t)), grad = grad_fn(step_rng, params, states, batch, state.step)
      grad = jax.lax.pmean(grad, axis_name='batch')
      new_optimizer = optimize_fn(state, grad)
      new_params_ema = jax.tree_map(
        lambda p_ema, p: p_ema * state.ema_rate + p * (1. - state.ema_rate),
        state.params_ema, new_optimizer.target
      )
      step = state.step + 1
      new_state = state.replace(
        step=step,
        optimizer=new_optimizer,
        model_state=new_model_state,
        params_ema=new_params_ema
      )
    else:
      loss, (_, losses_1, losses_2, losses_3, loss_t) = loss_fn(step_rng, state.params_ema, state.model_state, batch, 0)
      new_state = state

    loss = jax.lax.pmean(loss, axis_name='batch')
    losses_1 = jax.lax.pmean(losses_1, axis_name='batch')
    losses_2 = jax.lax.pmean(losses_2, axis_name='batch')
    losses_3 = jax.lax.pmean(losses_3, axis_name='batch')
    loss_t = jax.lax.pmean(loss_t, axis_name='batch')
    new_carry_state = (rng, new_state)
    return new_carry_state, (loss, losses_1, losses_2, losses_3, loss_t)

  return step_fn
