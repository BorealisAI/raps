""" Causal bandit definition. """
from abc import ABC, abstractmethod
from copy import deepcopy
from itertools import chain, combinations, permutations

import numpy as np

from raps.raps import iter_descendants
from raps.ucb import index_to_values


class Bandit(ABC):
  """ Abstract bandit class. """
  @property
  @abstractmethod
  def narms(self):
    """ The number of arms of this bandit. """

  @property
  @abstractmethod
  def batch_size(self):
    """ The number of parallel bandits with which interaction occcurs. """

  @abstractmethod
  def pull(self, arms):
    """ Pulls the arms of this bandit. """


def dfs(adj, node, time=0, times=None, visited=None):
  """ Depth first search in a graph. """
  if times is None:
    times = [-1 for _ in adj]
  if visited is None:
    visited = [False for _ in adj]
  visited[node] = True
  for child in reversed(adj[node]):
    if not visited[child]:
      time = dfs(adj, child, time, times=times, visited=visited)
  times[node] = time
  return time + 1


def topsort(adj):
  """ Returns topologically sorted list of nodes in the graph. """
  if not adj:
    return []
  ctime = 0
  times = [-1 for _ in adj]
  visited = [False for _ in adj]
  for node in range(len(adj)):
    if not visited[node]:
      ctime = dfs(adj, node, time=ctime, times=times, visited=visited)
  return sorted(list(range(len(adj))), key=times.__getitem__, reverse=True)


def revadj(adj):
  """ Returns reversed adjacency list (parents instead of children). """
  result = [[] for _ in adj]
  for node, children in enumerate(adj):
    for child in children:
      result[child].append(node)
  return result


def powerset(iterable):
  """powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"""
  items = list(iterable)
  return chain.from_iterable(combinations(items, r)
                             for r in range(len(items)+1))


class PCM:
  """ A probabilistic causal model. """
  # pylint: disable=too-many-arguments
  def __init__(self, adj, sems, probs=None, domain_size=2, batch_size=32):
    self.adj = adj
    self.revadj = revadj(adj)
    self.topsort = topsort(adj)
    self.sems = sems
    self.probs = probs
    self.domain_size = domain_size
    self.batch_size = batch_size
  # pylint: enable=too-many-arguments

  def sample(self):
    """ Samples variables from the graph. """
    sample = np.zeros([self.batch_size, len(self.adj)], dtype=int)
    for node in self.topsort:
      parents = self.revadj[node]
      vals = sample[:, parents].T
      sample[:, node] = self.sems[node](*vals)
    return sample

  def compute_probs(self):
    """ Computes observational probabilities of the model. """
    probs = np.zeros([len(self.adj), self.domain_size])
    for node in self.topsort:
      parents = self.revadj[node]
      probs[node] = self.probs[node](*probs[parents])
    return probs

  def intervene(self, mask, values):
    """ Samples variables from the graph after intervening. """
    if mask.shape != (self.batch_size, len(self.adj)):
      raise ValueError("mask.shape expected to be  "
                       f"{(self.batch_size, len(self.adj))=}, "
                       f"got {mask.shape=}")
    sample = np.zeros([self.batch_size, len(self.adj)], dtype=int)
    sample[mask] = values
    for node in self.topsort:
      batch_mask = ~mask[:, node]
      parents = self.revadj[node]
      sample[batch_mask, node] = self.sems[node](
          *sample[:, parents].T)[batch_mask]
    return sample

  def intervene_probs(self, mask, values):
    """ Computes probabilities in the intervened model. """
    if mask.shape != (len(self.adj),):
      raise ValueError(f"{mask.shape=} while expected {(len(self.adj),)}")
    probs = np.zeros([len(self.adj), self.domain_size])
    probs[mask, values] = 1
    for node in self.topsort:
      parents = self.revadj[node]
      if not mask[node]:
        probs[node] = self.probs[node](*probs[parents])
    return probs

  def obs_mean(self, node):
    """ Returns observational mean of a node. """
    return np.sum(self.compute_probs()[node] * np.arange(self.domain_size))

  def best_mean(self, node, return_parents=False, return_interventions=False):
    """ Returns best intervention mean of the node. """
    def ret(parents, interventions, mean):
      result = []
      if return_parents:
        result.append(parents)
      if return_interventions:
        result.append(interventions)
      result.append(mean)
      if len(result) == 1:
        return result[0]
      return result

    parents = self.revadj[node]
    if not parents:
      return ret(np.zeros(len(self.adj), dtype=bool),
                 None, np.sum(self.compute_probs()[node]
                              * np.arange(self.domain_size)))
    mask = np.zeros(len(self.adj), dtype=bool)
    mask[parents] = True
    best_mean = -float("inf")
    for index in range(self.domain_size ** len(parents)):
      values = index_to_values(np.asarray([index]),
                               np.asarray(parents),
                               self.domain_size)
      mean = np.sum(self.intervene_probs(mask, values)[node]
                    * np.arange(self.domain_size))
      if mean > best_mean:
        best_values = values
        best_mean = mean
    return ret(mask, best_values, mean)

  def iterate_intervene_probs(self, parents):
    """ Iterates over probabilities under all node and parents interventions. """
    parents = list(parents)
    for node in range(len(self.adj)):
      if node in parents:
        continue
      for index in range(self.domain_size ** len(parents)):
        mask = np.zeros(len(self.adj), dtype=bool)
        mask[np.asarray(parents, dtype=int)] = True
        values = index_to_values(np.asarray([index]), np.asarray(parents),
                                 self.domain_size)
        parents_probs = self.intervene_probs(mask, values)

        parents.append(node)
        mask[node] = True
        for value in range(self.domain_size):
          values = index_to_values(np.asarray([value * index]),
                                   np.asarray(parents),
                                   self.domain_size)
          node_probs = self.intervene_probs(mask, values)
          yield node, index, value, parents_probs, node_probs
        parents.pop()

  def eps(self, target_node, parents):
    """ Computes epsilon value for given node and subset of its parents. """
    result = (None, None, None, float("inf"))
    index_result = (None, None, None, float("inf"))
    for (node, index, value, parents_probs,
          node_probs) in self.iterate_intervene_probs(parents):
      if node in parents or node == target_node:
        continue
      descendants = np.asarray(
          [desc for desc in iter_descendants(self.adj, node)
           if desc not in [target_node, node] + parents])
      if descendants.size == 0:
        continue

      eps = np.min(np.max(np.abs(parents_probs[descendants]
                                 - node_probs[descendants]), 1), 0)
      if (index_result[:2] != (node, index)
          or index_result[0] == node and index_result[-1] < eps):
        if index_result[0] != node and index_result[-1] < result[-1]:
          result = index_result
        index_result = node, index, value, eps
    if index_result[-1] < result[-1]:
      result = index_result
    return result

  def gap(self, target_node, parents):
    """ Computes delta value for given node and subset of its' parents. """
    result = (None, None, None, float("inf"))
    index_result = (None, None, None, float("inf"))
    domain_vals = np.arange(self.domain_size)
    cut_revadj = deepcopy(self.revadj)
    for prnt in parents:
      cut_revadj[prnt].clear()
    for (node, index, value, parents_probs,
         node_probs) in self.iterate_intervene_probs(parents):
      if target_node not in iter_descendants(revadj(cut_revadj), node):
        continue
      gap = np.sum(np.abs((parents_probs[target_node]
                           - node_probs[target_node]) * domain_vals))
      if (index_result[:2] != (node, index)
          or index_result[0] == node and index_result[-1] < gap):
        if index_result[0] != node and index_result[-1] < result[-1]:
          result = index_result
        index_result = node, index, value, gap
    if index_result[-1] < result[-1]:
      result = index_result
    return result

  def min_eps_gap(self, node):
    """ Computes epsilon value of the PCM. """
    if node < 0:
      node = len(self.adj) + node
    def iter_subsets():
      for subset in powerset(self.revadj[node]):
        if (not subset
            or any(perm for perm in permutations(subset)
                   if not any(set(iter_descendants(self.adj, prnt))
                              & set(perm[i + 1:])
                              for i, prnt in enumerate(perm)))):
          yield subset
    return (min(self.eps(node, list(subset))[-1]
                for subset in iter_subsets()),
            min(self.gap(node, list(subset))[-1]
                for subset in iter_subsets()))

class CausalBandit(Bandit):
  """ Bandit with causal structure over the arms. """
  def __init__(self, pcm, arm_variables=None, reward_node=None):
    self.pcm = pcm
    if arm_variables is None:
      arm_variables = list(range(len(pcm.adj) - 1))
    self.arm_variables = np.asarray(arm_variables)
    if reward_node is None:
      reward_node = len(self.pcm.adj) - 1
    self.reward_node = reward_node
    self.timestep = 0

  @property
  def narms(self):
    return self.arm_variables.size * self.pcm.domain_size + 1

  @property
  def batch_size(self):
    return self.pcm.batch_size

  def pull_atomic_intervention(self, arms):
    """ Performs arm pull for atomic intervention arms. """
    intervention_mask = np.zeros((self.batch_size, len(self.pcm.adj)),
                                 dtype=bool)
    arms_mask = arms != self.narms - 1
    values = arms[arms_mask] % self.pcm.domain_size
    variables_index = (arms[arms_mask]
                       // self.pcm.domain_size) % self.arm_variables.size
    batch_range = np.arange(self.batch_size)
    intervention_mask[batch_range[arms_mask],
                      self.arm_variables[variables_index]] = True
    sample = self.pcm.intervene(intervention_mask, values)
    return sample

  def pull_multi_intervention(self, arms):
    """ Pulls arms with possibly multi-node intervention. """
    mask = arms != self.pcm.domain_size
    return self.pcm.intervene(mask, arms[mask])

  def pull(self, arms):
    if arms.shape == (self.batch_size,):
      return self.pull_atomic_intervention(arms)
    assert arms.shape == (self.batch_size, len(self.pcm.adj)), arms.shape
    return self.pull_multi_intervention(arms)
