# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
""" Plot utilities. """
import os
from collections import defaultdict
from contextlib import contextmanager

import matplotlib.pyplot as plt

from raps.exp import load
from u4ml.plot import LinesPlotter, MeansPlotter, plot_mean_std, refresh_axis


class RegretPlotter:
  """ Plots regret. """
  def __init__(self, horizon, plot_period, lines_plotter=None):
    self.horizon = horizon
    self.plot_period = plot_period
    self.lines_plotter = (lines_plotter if lines_plotter is not None
                          else LinesPlotter(ax=plt.gca()))
    self.timestep = None
    self.run_index = None
    self.should_plot_flag = None
    self.lines = defaultdict(list)

  @classmethod
  @contextmanager
  def make_autoclear_context(cls, horizon, plot_period, ax=None, output=None):
    """ Creates an instance and clears context upon exiting the context. """
    with LinesPlotter.make_autoclear_context(ax=ax, output=output) as plotter:
      instance = cls(horizon, plot_period, lines_plotter=plotter)
      try:
        yield instance
      finally:
        pass

  def should_plot(self, alg, key, regrets, run_index):
    """ Returns whether or not it is time to plot. """
    if run_index != self.run_index:
      self.should_plot_flag = True
      return self.should_plot_flag
    timestep = len(regrets[key][-1])
    if timestep == self.timestep and alg.name != "rapsucb":
      return timestep % self.plot_period == 0 or self.should_plot_flag
    self.should_plot_flag = (timestep % self.plot_period == 0
                             or timestep == self.horizon)
    return self.should_plot_flag

  def plot(self, alg, key, regrets, run_index):
    """ Performs plotting at the end of a round. """
    plt.sca(self.lines_plotter.ax)
    if not self.should_plot(alg, key, regrets, run_index):
      return
    self.timestep = len(regrets[key][-1])
    self.lines_plotter.extend(key, [self.timestep - 1],
                              [regrets[key][-1][-1].mean(0)])
    line_len = len(self.lines_plotter.lines[key].get_data()[1])
    min_line_len = min(len(line.get_data()[1])
                       for line in self.lines_plotter.lines.values())
    if self.timestep == self.horizon and line_len == min_line_len:
      for child in self.lines_plotter.ax.get_children():
        try:
          child.remove()
        except NotImplementedError:
          pass
      for linekey in list(self.lines_plotter.lines):
        line = self.lines_plotter.lines[linekey]
        self.lines[linekey].append(line)
        plot_mean_std(self.lines[linekey][0].get_data()[0],
                      [line.get_data()[1] for line in self.lines[linekey]],
                      color=line.get_color())
        self.lines_plotter.lines.pop(linekey)
        self.lines_plotter.plot_line(linekey, [], [], color=line.get_color())
      refresh_axis()
    self.run_index = run_index


class FigMeansPlotter(MeansPlotter):
  """ Mans of figures loaded from files plotter. """
  @classmethod
  def from_dir(cls, dirpath, keyformat="{key}",
               file_pattern="{run_index:02d}-fig.pickle",
               nruns=10, **kwargs):
    """ Creates an instance by loading figures from directory. """
    lines_plotter = LinesPlotter(**kwargs)
    plt.close() # close the plot to prevent jupyter render
    result = cls(lines_plotter)
    for i in range(nruns):
      fig = load(os.path.join(dirpath, file_pattern.format(run_index=i)))
      plt.close()
      for line in fig.get_axes()[0].lines:
        result.lines[keyformat.format(key=line.get_label())].append(line)
    return result

  def show(self):
    """ Shows the axis. """
    plt.sca(self.lines_plotter.ax)
    plt.show()
