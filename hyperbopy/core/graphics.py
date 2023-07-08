import matplotlib.pyplot as plt

color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']


class SimuFigure:

    def __init__(self, W0, var_names, x, Z, dN_fig=200):
        self.W0 = W0
        self.var_names = var_names
        self.x = x
        self.Z = Z
        self.dN_fig = dN_fig if dN_fig is not None else 200
        #
        self.fig = None
        self.axarr = None
        self.lines = None
        self.init_plot()

    def plot_allvar(self, W):
        self.lines = [ax.plot(self.x, w, color=color)[0]
                      for ax, w, color in zip(self.axarr.flatten(), W, color_cycle)]

    def clearlines(self):
        for l in self.lines:
            l.remove()

    def update_plot(self, W, title=None):
        self.clearlines()
        self.plot_allvar(W)
        if title is not None:
            plt.suptitle(title)
        plt.pause(0.005)

    def init_plot(self):
        self.fig, self.axarr = plt.subplots(
            self.W0.shape[0], 1, constrained_layout=True, sharex=True)
        #
        self.axarr.flatten()[-1].set_xlabel('Horizontal coordinate [m]')
        for i, (ax, var_label) in enumerate(zip(self.axarr.flatten(), self.var_names)):
            ax.set_ylabel(var_label)
        #
        self.plot_allvar(self.W0)
