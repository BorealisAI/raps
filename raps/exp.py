""" Experiment definitions. """
import os
import pickle
from collections import OrderedDict, defaultdict
from inspect import signature

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import trange

from raps.bandit import PCM, CausalBandit
from raps.raps import (RAPSUCB, make_erdos_renyi,
                       find_all_targets, iter_descendants)
from raps.ucb import UCB
from u4ml.plot import plot_mean_std


class UniformPCMFuncs:
  """ Definitions of PCM functions. """
  def __init__(self, domain_size=4, cliprange=(0.25, 0.75),
               batch_size=1, noise_prob=0.1):
    self.domain_size = domain_size
    self.cliprange = cliprange
    self.batch_size = batch_size
    self.noise_prob = noise_prob

  def sem_func(self, *parents):
    """ Structural equation model function. """
    if not parents:
      return np.random.randint(self.domain_size, size=self.batch_size)
    assert 0 <= np.min(parents) and np.max(parents) < self.domain_size, parents
    result = np.stack(parents)[np.random.randint(len(parents),
                                                 size=self.batch_size),
                               np.arange(self.batch_size)]
    mask = np.random.random(self.batch_size) < self.noise_prob
    result[mask] = np.random.randint(self.domain_size, size=np.sum(mask))
    return result

  def reward_func(self, *parents):
    """ Reward function. """
    prob = np.mean(parents, 0) / (self.domain_size - 1)
    return np.random.random(self.batch_size) < np.clip(prob, *self.cliprange)

  def prob_func(self, *parents_probs):
    """ Probability of samples function. """
    if not parents_probs:
      return np.full(self.domain_size, 1 / self.domain_size)
    parents_probs = np.stack(parents_probs)
    num_parents = parents_probs.shape[0]
    parents_dist = np.full(num_parents, 1 / num_parents)
    return ((1 - self.noise_prob)
            * np.sum(parents_probs * parents_dist[:, None], 0)
            + self.noise_prob / self.domain_size)

  def reward_prob(self, *parents):
    """ Reward probability function. """
    prob = np.clip(np.sum(np.arange(self.domain_size) * parents)
                   / (self.domain_size - 1) / len(parents), *self.cliprange)
    return np.asarray([1 - prob, prob] + [0] * (self.domain_size - 2))

  def sems_list(self, num_nodes, reward_node=None):
    """ Returns list of SEM funcs. """
    if reward_node is None:
      reward_node = num_nodes - 1
    return [self.sem_func if node != reward_node
            else self.reward_func for node in range(num_nodes)]

  def probs_list(self, num_nodes, reward_node=None):
    """ Returns list of probability functions. """
    if reward_node is None:
      reward_node = num_nodes - 1
    return [self.prob_func if node != reward_node
            else self.reward_prob for node in range(num_nodes)]


def dump(obj, filepath):
  """ Saves object to filepath using pickle. """
  with open(filepath, "wb") as dumpfile:
    pickle.dump(obj, dumpfile)

def load(filepath):
  """ Loads object from file using pickle. """
  with open(filepath, "rb") as loadfile:
    return pickle.load(loadfile)


class Exp:
  """ Experiment class. """
  def __init__(self, bandit, algs, **kwargs):
    self.bandit = bandit
    self.algs = algs
    self.kwargs = kwargs
    self.regrets = defaultdict(list)

  @staticmethod
  def make_tree(num_nodes, num_parents=1):
    """ Creates adjecency lists for tree graph. """
    adj = []
    num_nodes -= 1
    for node in range(1, num_nodes, 2):
      adj.append([node])
      if node + 1 < num_nodes:
        adj[-1].append(node + 1)
    while num_nodes - 1 >= len(adj):
      adj.append([])

    parents = np.random.choice(num_nodes, replace=False, size=num_parents)
    for prnt in parents:
      adj[prnt].append(num_nodes)
    adj.append([])
    return adj

  @staticmethod
  def make_erdos_renyi(num_nodes, prob, num_parents=1):
    """ Creates Erdos-Renyi graph. """
    adj = make_erdos_renyi(num_nodes - 1, prob)
    parents = np.random.randint(num_nodes - 1, size=num_parents)
    for prnt in parents:
      adj[prnt].append(num_nodes - 1)
    adj.append([])
    return adj

  @staticmethod
  def make_bandit(adj, reward_node=None, domain_size=4,
                  cliprange=(0, 1), noise_prob=0.1):
    """ Creates an instance with uniform sem funcs. """
    funcs = UniformPCMFuncs(domain_size=domain_size,
                            cliprange=cliprange,
                            noise_prob=noise_prob)
    pcm = PCM(adj, sems=funcs.sems_list(len(adj), reward_node),
              probs=funcs.probs_list(len(adj), reward_node),
              domain_size=funcs.domain_size,
              batch_size=funcs.batch_size)
    bandit = CausalBandit(pcm=pcm, reward_node=reward_node)
    return bandit

  @classmethod
  def make(cls, adj=None, bandit=None, algclasses=None, **kwargs):
    """ Creates instance of the experiment from bandit and alg classes. """
    if callable(adj) and bandit is None:
      kwargs["adj"] = adj
      adj = adj(**{key: val for key, val in kwargs.items()
                   if key in signature(adj).parameters})
    if bandit is None:
      bandit = cls.make_bandit(
          adj=adj, **{key: val for key, val in kwargs.items()
                      if key in signature(cls.make_bandit).parameters
                      and key != "adj"})
    if algclasses is None:
      algclasses = [RAPSUCB, UCB]
    algs = [
        algcls.from_pcm(bandit.pcm, reward_node=bandit.reward_node, **{
            key: val for key, val in kwargs.items()
            if key in signature(algcls.from_pcm).parameters})
        for algcls in algclasses
    ]
    return cls(bandit, OrderedDict((alg.name, alg) for alg in algs), **kwargs)

  @classmethod
  def from_dir(cls, directory,
               file_pattern="{run_index:02d}-exp.pickle",
               nruns=10):
    """ Creates an instance from directory of runs. """
    result = None
    for run_index in range(nruns):
      exp = load(os.path.join(directory,
                              file_pattern.format(run_index=run_index)))
      if result is None:
        result = exp
      else:
        for key in exp.regrets:
          result.regrets[key].extend(exp.regrets[key])
    return result

  def iterun(self, horizon, leave_tqdm=True):
    """ Iterate over a run of the experiment. """
    best_mean = self.bandit.pcm.best_mean(self.bandit.reward_node)
    for key in self.algs:
      self.regrets[key].append([np.zeros(self.bandit.batch_size)])
    for _ in trange(horizon, leave=leave_tqdm):
      for key, alg in self.algs.items():
        arms = alg.get_arms()
        rewards = self.bandit.pull(arms)
        alg.update_stats(arms, rewards)
        self.regrets[key][-1].append(
            self.regrets[key][-1][-1]
            + best_mean - rewards[..., self.bandit.reward_node]
        )
        yield alg, key, self.regrets

  def single_run(self, horizon, leave_tqdm=True, plotters=None, run_index=None):
    """ Perform a single run of the experiment. """
    if not plotters:
      plotters = []
    for alg, key, regrets in self.iterun(horizon, leave_tqdm):
      for pltr in plotters:
        pltr.plot(alg, key, regrets, run_index=run_index)

  def run(self, ntimes, horizon, plotters=None):
    """ Runs the experiment specified number of times for given horizon. """
    for i in trange(ntimes):
      self.single_run(horizon, leave_tqdm=False,
                      plotters=plotters, run_index=i)
      if i != ntimes - 1:
        new_exp = self.make(**self.kwargs)
        self.bandit = new_exp.bandit
        self.algs = new_exp.algs

  def plot(self, ax=None, npoints=100, linewidth=4,
           plot_fn=plot_mean_std, **kwargs):
    """ Plots the results of this experiment. """
    horizon = len(next(iter(self.regrets.values()))[0])
    plot_period = int(horizon / npoints)
    xrange = np.arange(0, horizon, plot_period)
    if ax is not None:
      plt.sca(ax)
    for key, regrets in self.regrets.items():
      plot_fn(xrange, np.squeeze(regrets, -1).T[::plot_period],
              axis=1, label=key, lw=linewidth, **kwargs)
    plt.xlabel(r"$T$")
    plt.ylabel(r"Regret")
    return plt.gcf()


class MultiParentNumInterventionsExp:
  """ Verify that number of interventions in multiparent case matches theory.
  """
  def __init__(self, num_nodes=None, num_parents=(5, 10, 20),
               colors=("C2", "C1", "C3"), plot_bounds=True):
    if num_nodes is None:
      num_nodes = np.logspace(5, 10, base=2, num=6).astype(int)
    self.num_nodes = num_nodes
    self.num_parents = num_parents
    self.colors = colors
    self.bounds = {}
    if plot_bounds:
      self.plot_bounds()

  def plot_bounds(self,
                  label_format=r"${factor}\log_2(n)$",
                  linewidth=4):
    """ Plots the theoretical bounds. """
    for i, num_parents in enumerate(self.num_parents):
      self.bounds[num_parents] = plt.plot(
          self.num_nodes,
          (num_parents + 1) * np.log2(self.num_nodes),
          label=label_format.format(factor=num_parents + 1),
          color=self.colors[i], linewidth=linewidth)

  def sample(self, num_nodes):
    """ Samples data for experiment with specified num_nodes. """
    loglogn = np.log(max(np.e ** 2, np.log(num_nodes)))
    prob = 1 - (0.5 / (loglogn - 1)) ** (1 / (loglogn - 1))
    adj, perm = make_erdos_renyi(num_nodes, prob, return_perm=True)
    parents = np.random.choice(num_nodes, size=self.num_parents[-1],
                               replace=False)
    return adj, perm, parents

  def iterun_adj(self, adj, perm, parents):
    """ Finds for parents in specified adjacency list. """
    parents = sorted(parents, key=perm.__getitem__, reverse=True)
    prev_num_parents = 0
    num_interventions = 0
    candidates = list(range(len(adj)))
    for num_parents in self.num_parents:
      num_parents -= prev_num_parents
      (num_new_intervnetions,
       num_no_parents) = find_all_targets(adj, perm, parents[:num_parents],
                                          candidates)
      num_interventions += num_new_intervnetions
      candidates = [cand for cand in candidates
                    if cand not in set(
                        desc for prnt in parents[:num_parents]
                        for desc in iter_descendants(adj, prnt)
                    )]
      parents = parents[num_parents:]
      yield num_parents + prev_num_parents, num_interventions + num_no_parents
      prev_num_parents += num_parents

  def initialize_colors(self, plotter,
                        label_format=r"RAPS, $|\mathcal{{P}}|$={num_parents}",
                        redraw_legend=False, **kwargs):
    """ Initializes colors for lines of the experiment. """
    for i, num_parents in enumerate(self.num_parents):
      label = label_format.format(num_parents=num_parents)
      plotter.plot_line(label, [], [], color=self.colors[i],
                        redraw_legend=redraw_legend, **kwargs)

  def run(self, plotter, label_format=r"RAPS, $|\mathcal{{P}}|$={num_parents}"):
    """ Runs the experiment. """
    for num_nodes in self.num_nodes:
      adj, perm, parents = self.sample(num_nodes)
      for num_parents, num_interventions in self.iterun_adj(adj, perm, parents):
        plotter.extend(label_format.format(num_parents=num_parents),
                       [num_nodes], [num_interventions], redraw_legend=False)
