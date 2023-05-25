# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
""" UCB-based algorithms. """
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


class CausalBanditAlg(ABC):
  """ Abstract causal bandit algorithm. """
  def __init__(self, num_nodes, domain_size=2, batch_size=20, reward_node=-1):
    self.num_nodes = num_nodes
    self.domain_size = domain_size
    self.batch_size = batch_size
    self.reward_node = (reward_node if reward_node >= 0
                        else num_nodes + reward_node)

  @property
  def name(self):
    """ Returns the name of the alg. """
    return self.__class__.__name__.lower()

  @abstractmethod
  def get_arms(self):
    """ Returns arms to be pulled when interacting with the bandit. """

  @abstractmethod
  def update_stats(self, arms, rewards):
    """ Updates the internal statistic of the algorithm. """


@dataclass
class UCBStats:
  """ Statistics of a UCB-type algorithm. """
  means: np.array
  conf_bounds: np.array
  npulls: np.array

  def __init__(self, narms, batch_size=100):
    self.means = np.zeros([batch_size, narms], dtype=float)
    self.conf_bounds = np.zeros([batch_size, narms], dtype=float)
    self.npulls = np.zeros([batch_size, narms], dtype=int)

  def reset(self):
    """ Resets the statistics. """
    self.means.fill(0)
    self.conf_bounds.fill(0)
    self.npulls.fill(0)


def index_to_values(index, arm_variables, domain_size=2):
  """ Converts index of arm to intervention values. """
  result = np.zeros((index.shape[0], len(arm_variables)), dtype=int)
  index = np.copy(index)
  for i in range(len(arm_variables) - 1, -1, -1):
    result[..., i] = index % domain_size
    index //= domain_size
  return result

def values_to_index(values, arm_variables, domain_size=2):
  """ Converts values of interventions to index of arm. """
  powers = domain_size ** np.arange(len(arm_variables) - 1, -1, -1)
  return np.sum(values[..., arm_variables] * powers, -1)


class UCB(CausalBanditAlg):
  """ UCB with interventions of size K^n. """
  def __init__(self, arm_variables, **kwargs):
    super().__init__(**kwargs)
    if arm_variables is None:
      mask = np.ones((self.num_nodes), dtype=bool)
      mask[self.reward_node] = False
      arm_variables, = np.where(mask)
    self.arm_variables = arm_variables
    self.stats = UCBStats(self.narms, self.batch_size)
    self.timestep = 0

  @classmethod
  def from_pcm(cls, pcm, reward_node=None):
    """ Creates an instance from PCM. """
    return cls(arm_variables=None,
               num_nodes=len(pcm.adj),
               reward_node=reward_node,
               domain_size=pcm.domain_size,
               batch_size=pcm.batch_size)

  @property
  def narms(self):
    """ Returns the number of arms of the bandit this algo interacts with. """
    return self.domain_size ** len(self.arm_variables)

  def get_arms(self):
    arms = np.full((self.batch_size, self.num_nodes), self.domain_size)
    if self.timestep < self.narms:
      index = np.full(self.batch_size, self.timestep)
    else:
      index = np.argmax(self.stats.conf_bounds, -1)
    arms[:, self.arm_variables] = index_to_values(index, self.arm_variables,
                                                  self.domain_size)
    return arms

  def update_stats(self, arms, rewards):
    arms_index = values_to_index(arms, self.arm_variables, self.domain_size)
    batch_index = np.arange(self.batch_size)
    npulls = self.stats.npulls[batch_index, arms_index]
    self.stats.means[batch_index, arms_index] = (
        npulls / (npulls + 1) * self.stats.means[batch_index, arms_index]
        + rewards[..., self.reward_node] / (npulls + 1)
    )
    self.stats.conf_bounds[batch_index, arms_index] = (
        self.stats.means[batch_index, arms_index]
        + np.sqrt(2 * np.log(self.timestep + 1) / (npulls + 1))
    )
    self.stats.npulls[batch_index, arms_index] += 1
    self.timestep += 1
