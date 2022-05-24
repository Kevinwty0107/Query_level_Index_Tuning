import os
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt


class RewardPlotter(object):
    
    def __init__(self, plot_name):
        self.fig = plt.figure()
        self.plot_name = plot_name
        self.epochs = []
        self.rewards = []
    
    def new_figure(self):
        self.fig = plt.figure()

    def plot(self, epoch, reward):
        self.epochs.append(epoch)
        self.rewards.append(reward)
        plt.cla()
        plt.plot(self.epochs, self.rewards, marker='+')
        self.fig.savefig(self.plot_name)
        plt.pause(0.000001)

    def plot_list(self, epochs, rewards, xlabel=None, ylabel=None):
        self.fig = plt.figure()
        plt.plot(epochs, rewards, marker='+')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        self.fig.savefig(self.plot_name)

    def hold_plot(self):
        self.fig.savefig(self.plot_name)
        plt.pause(100000000)
