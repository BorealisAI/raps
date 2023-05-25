""" RAPS algorithm-related definitions. """
from dataclasses import dataclass, field
from math import ceil, log
from random import choice, random, shuffle
from typing import List, Union

import numpy as np

from raps.ucb import UCB, CausalBanditAlg, index_to_values


def permutation(num):
  """ Permutation of num elements. """
  result = list(range(num))
  shuffle(result)
  return result

def inverse_permutation(perm):
  """ Computes the inverse of a given permutation. """
  result = [0 for _ in perm]
  for i in range(len(result)):
    result[perm[i]] = i
  return result


def make_erdos_renyi(num_vertices, prob, return_perm=False):
  """ Returns DAG sampled according to Erdos-Renyi model. """
  perm = permutation(num_vertices)
  adj = [[] for _ in range(num_vertices)]
  for i in range(num_vertices - 1):
    for j in range(i + 1, num_vertices):
      if random() > prob:
        continue
      if perm[i] > perm[j]:
        adj[j].append(i)
      else:
        adj[i].append(j)
  if return_perm:
    return adj, perm
  return adj


def iter_descendants(adj, node):
  """ Iterator over descendants of the given node. """
  if node is None:
    return
  visited = set()

  def rec(node):
    if node in visited:
      return
    yield node
    visited.add(node)
    for child in adj[node]:
      yield from rec(child)

  yield from rec(node)


def transpose(adj):
  """ Returns the adjacency list for the transposed graph. """
  result = [[] for _ in adj]
  for i in range(len(adj)):
    for j in adj[i]:
      result[j].append(i)
  return result


def iter_ancestors(adj, node):
  """ Iterator over ancestors of the given node. """
  yield from iter_descendants(transpose(adj), node)


def find_target(adj, target, candidates=None):
  """ Finds target in a DAG specified by adjacency list. """
  ancestors = set(iter_ancestors(adj, target))
  if candidates is None:
    candidates = list(range(len(adj)))
  num_interventions = 0

  def rec(candidates):
    nonlocal num_interventions
    if not candidates:
      return None
    sample = choice(candidates)
    num_interventions += 1
    descendants = set(iter_descendants(adj, sample)) & set(candidates)
    if sample in ancestors:
      descendants.remove(sample)
      result = rec(list(descendants))
      if result is None:
        return sample
      return result
    return rec([node for node in candidates if node not in descendants])

  found = rec(candidates)
  assert found == target, (found, target)
  return num_interventions


def find_all_targets(adj, perm, targets, candidates=None):
  """ Finds a set of targets. """
  if candidates is None:
    candidates = list(range(len(adj)))
  targets = sorted(targets, key=perm.__getitem__, reverse=True)
  result = 0
  for trgt in targets:
    result += find_target(adj, trgt, candidates)
    candidates = [cand for cand in candidates
                  if cand not in set(iter_descendants(adj, trgt))]
  return result, find_target(adj, None, candidates)


def lower_bound(adj, target=None):
  """ Computes the lower bound on the number of interventions required. """
  adj_transpose = transpose(adj)
  target_ancestors = (set() if target is None
                      else set(iter_descendants(adj_transpose, target)))
  result = 0
  for node in range(len(adj)):
    result += 1 / (1 + len((set(iter_descendants(adj_transpose, node))
                            ^ target_ancestors) - {node}))
  return result


@dataclass
class RAPSHistory:
  """ Raps interaction history. """
  observations: Union[List[np.ndarray], np.ndarray] \
      = field(default_factory=list)
  interventions: Union[List[np.ndarray], np.ndarray] \
      = field(default_factory=list)


@dataclass
class LastAncestorsStats:
  """ Statistics relative to last ancestor discovered by RAPS. """
  ancestors: np.ndarray
  descendants: np.ndarray
  history: RAPSHistory


# pylint: disable=too-many-instance-attributes
@dataclass
class RAPSStats:
  """ RAPS algorithm statistics. """
  candidates: np.ndarray
  banned_candidates: np.ndarray
  nodes: Union[None, np.ndarray]
  last_ancestors_stats: Union[None, LastAncestorsStats]
  parents: np.ndarray
  nodes_value: Union[None, int]
  parents_value_index: Union[None, int]
  history: RAPSHistory

  def __init__(self, batch_size, num_nodes, reward_node):
    self.candidates = np.ones((batch_size, num_nodes), dtype=bool)
    self.candidates[:, reward_node] = False
    self.banned_candidates = np.zeros((batch_size, num_nodes), dtype=bool)
    self.nodes = None
    self.last_ancestors_stats = None
    self.parents = np.zeros((batch_size, num_nodes), dtype=bool)
    self.nodes_value = None
    self.parents_value_index = None
    self.history = RAPSHistory()

  def reset_candidates(self, reward_node):
    """ Resets candidate nodes. """
    self.candidates.fill(True)
    self.candidates[:, reward_node] = False
    self.candidates[self.parents] = False
    self.candidates[self.banned_candidates] = False
# pylint: enable=too-many-instance-attributes


class RAPSUCB(CausalBanditAlg):
  """ Randomized parent search algorithm + upper confidence bound algorithm. """
  def __init__(self, eps, gap, delta, **kwargs):
    super().__init__(**kwargs)
    self.gap = gap
    self.delta = delta
    self.eps = eps
    self.budget = self.compute_budget(eps, gap, delta, self.num_nodes,
                                      self.domain_size)
    self.stats = RAPSStats(self.batch_size, self.num_nodes, self.reward_node)
    self.ucb = None
    self.timestep = 0

  @staticmethod
  def compute_budget(eps, gap, delta, num_nodes, domain_size=2):
    """ Computes the budget. """
    return max(ceil(32 * log(8 * num_nodes
                             * domain_size
                             * (domain_size + 1) ** num_nodes
                             / delta)
                    / gap ** 2),
               ceil(8 * log(8 * num_nodes ** 2
                            * domain_size ** 2
                            * (domain_size + 1) ** num_nodes
                            / delta) / eps ** 2))

  @classmethod
  def from_pcm(cls, pcm, reward_node=None, delta=0.01):
    """ Creates an instance from PCM. """
    if reward_node is None:
      reward_node = len(pcm.adj) - 1
    eps, gap = pcm.min_eps_gap(reward_node)
    return cls(eps, gap, delta,
               num_nodes=len(pcm.adj),
               reward_node=reward_node,
               domain_size=pcm.domain_size,
               batch_size=pcm.batch_size)

  def get_arms(self):
    if self.timestep < self.budget:
      return np.full((self.batch_size, self.num_nodes), self.domain_size)
    if self.ucb is not None:
      return self.ucb.get_arms()
    arms = np.full((self.batch_size, self.num_nodes), self.domain_size)
    arms[np.arange(self.batch_size), self.stats.nodes] = self.stats.nodes_value
    if self.stats.parents_value_index is not None:
      arms[self.stats.parents] = np.reshape(
          index_to_values(self.stats.parents_value_index,
                          np.where(self.stats.parents)[1],
                          self.domain_size),
          -1)
    return arms

  def is_at_reward_ancestor(self):
    """ Returns true if the current node is reward ancestor. """
    assert self.should_update_nodes(), (len(self.stats.history.interventions),
                                        self.budget)
    observations, interventions = map(
        np.asarray, (self.stats.history.observations,
                     self.stats.history.interventions))
    # reshape: time x batch x num_nodes ->
    #   -> nodes_domain x parents_domain x budget x batch
    parents_domain = np.squeeze(
        self.domain_size ** np.sum(self.stats.parents, 1))
    observations = np.reshape(observations[..., self.reward_node],
                              (1, parents_domain, self.budget, self.batch_size))
    obs_reward = np.mean(observations, 2)
    interventions = np.reshape(interventions[..., self.reward_node],
                               (self.domain_size, parents_domain,
                                self.budget, self.batch_size))
    interv_reward = np.mean(interventions, 2)
    return np.any(np.abs(obs_reward - interv_reward)
                  > self.gap / 2, axis=(0, 1))

  def descendants(self):
    """ Returns estimated descendants of the current node. """
    assert self.should_update_nodes(), (len(self.stats.history.interventions),
                                        self.budget)
    observations = np.asarray(self.stats.history.observations)
    # reshape: time x batch x num_nodes ->
    #   -> nodes_domain x parents_domain x budget x batch x num_nodes
    parents_domain = np.squeeze(
        self.domain_size ** np.sum(self.stats.parents, 1))
    observations = np.reshape(observations,
                              (1, parents_domain, self.budget,
                               self.batch_size, self.num_nodes))
    observations = np.mean(observations, 2)
    interventions = np.reshape(
        np.asarray(self.stats.history.interventions),
        (self.domain_size, parents_domain, self.budget,
         self.batch_size, self.num_nodes))
    interventions = np.mean(interventions, 2)
    # nodes_domain x parents_domain x batch x num_nodes
    result = np.any(
        np.abs(observations - interventions)
        > self.eps / 2, axis=(0, 1))
    result[:, self.reward_node] = False
    result[self.stats.parents] = False
    return result

  def should_update_nodes(self):
    """ Returns true if it is time to update the intervened on node. """
    num_parents = np.squeeze(np.sum(self.stats.parents, 1))
    num_interventions = len(self.stats.history.interventions)
    return (num_interventions
            and (num_interventions
                 % (self.domain_size ** (num_parents + 1) * self.budget) == 0))

  def should_update_nodes_value(self):
    """ Returns true if it is time to update the domain value. """
    num_parents = np.sum(self.stats.parents, -1)
    num_interventions = len(self.stats.history.interventions)
    return (num_interventions > 0
            and (num_interventions
                 % (self.domain_size ** num_parents * self.budget) == 0))

  def should_update_parents_value_index(self):
    """ Returns true if parents value index should be updated. """
    num_interventions = len(self.stats.history.interventions)
    return num_interventions > 0 and num_interventions % self.budget == 0

  def sample_nodes(self):
    """ Samples nodes. """
    return np.asarray([np.random.choice(*np.where(candidates))
                       for candidates in self.stats.candidates])

  def update_stats(self, arms, rewards):
    self.timestep += 1
    if self.ucb is not None:
      self.ucb.update_stats(arms, rewards)
      return
    intervened_mask = arms != self.domain_size
    if self.timestep <= self.budget:
      assert not np.any(intervened_mask), intervened_mask
      self.stats.history.observations.append(rewards)
      if self.timestep == self.budget:
        self.stats.nodes_value = 0
        self.stats.nodes = self.sample_nodes()
      return

    assert np.sum(intervened_mask, 1) == np.sum(self.stats.parents, 1) + 1,\
        (intervened_mask, self.stats.parents, self.stats.nodes)
    self.stats.history.interventions.append(rewards)
    if self.should_update_nodes_value():
      self.stats.nodes_value = (self.stats.nodes_value + 1) % self.domain_size
    if self.should_update_parents_value_index():
      self.stats.parents_value_index = (
          ((self.stats.parents_value_index or 0) + 1)
          % (self.domain_size ** np.sum(self.stats.parents, 1))
      )
    if not self.should_update_nodes():
      return
    descendants = self.descendants()
    if np.squeeze(self.is_at_reward_ancestor()):
      descendants[np.arange(self.batch_size), self.stats.nodes] = False
      self.stats.candidates &= descendants
      self.stats.last_ancestors_stats = LastAncestorsStats(
          np.copy(self.stats.nodes), descendants,
          RAPSHistory(interventions=np.array(self.stats.history.interventions)))
    else:
      self.stats.candidates[descendants] = False
      self.stats.banned_candidates[descendants] = True

    if np.squeeze(np.any(self.stats.candidates, 1)):
      self.stats.history.interventions.clear()
      self.stats.nodes = self.sample_nodes()
      return
    if self.stats.last_ancestors_stats is not None:
      self.stats.banned_candidates[
          self.stats.last_ancestors_stats.descendants] = True
      assert not np.any(
          self.stats.parents[np.arange(self.batch_size),
                             self.stats.last_ancestors_stats.ancestors]),\
          (self.stats.parents, self.stats.last_ancestors_stats)
      self.stats.parents[np.arange(self.batch_size),
                         self.stats.last_ancestors_stats.ancestors] = True
      self.stats.history.observations \
          = self.stats.last_ancestors_stats.history.interventions

    self.stats.reset_candidates(self.reward_node)
    if (self.stats.last_ancestors_stats is None
        or not np.squeeze(np.any(self.stats.candidates, 1))):
      self.ucb = UCB(np.where(self.stats.parents)[1],
                     num_nodes=self.num_nodes,
                     domain_size=self.domain_size,
                     batch_size=self.batch_size,
                     reward_node=self.reward_node)
    else:
      self.stats.nodes = self.sample_nodes()
      self.stats.last_ancestors_stats = None
      self.stats.history.interventions.clear()
