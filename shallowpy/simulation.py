import matplotlib.pyplot as plt
import numpy as np

from shallowpy.core.spatial_scheme import available_spatial_schemes
from shallowpy.core.temporal_schemes import available_temporal_schemes
from shallowpy.core.graphics import SimuFigure

# ### main run function


class Simulation:

    def __init__(self, model, W0, dx, temporal_scheme=None, spatial_scheme=None, dt_fact=None):
        self.model = model
        self.temporal_scheme_name = temporal_scheme if temporal_scheme is not None else 'RungeKutta33'
        self.spatial_scheme_name = spatial_scheme if spatial_scheme is not None else 'CentralUpwind'
        self.W0 = W0
        self.dx = dx
        self.dt_fact = dt_fact if dt_fact is not None else 0.5
        #
        self.euler_step = 'nonhydro' if self.spatial_scheme_name == 'CentralUpwindPathNoneHydro' else 'regular'
        self.SpatialScheme = available_spatial_schemes[self.spatial_scheme_name](
            self.model)
        self.TemporalScheme = available_temporal_schemes[self.temporal_scheme_name](
            self.model, self.SpatialScheme, self.dt_fact)

    def run_simulation(self, tmax, plot_fig=True, dN_fig=200, dt_save=None, x=None, Z=None):
        #
        dt_save = dt_save if dt_save is not None else tmax/100
        #
        # Initialization
        W = np.copy(self.W0)
        t = 0  # time tracking
        Nt = 0  # time steps
        #
        U_save = [W[:-1, :]]
        t_save = [0]
        #
        if plot_fig:
            simu_fig = SimuFigure(self.W0, self.model.var_names, x, Z, dN_fig)
        # Running simulation
        while t <= tmax:
            # update time step
            W, dt = self.TemporalScheme.time_step(
                W, self.dx, euler_step=self.euler_step)
            t += dt
            Nt += 1
            # update saved time steps
            if (t - t_save[-1]) >= dt_save:
                t_save.append(t)
                U_save.append(W[:-1, :])
            # update interactive figure
            if plot_fig & (Nt % dN_fig == 0):
                simu_fig.update_plot(W, title='{:.1e} s, {:0d}'.format(t, Nt))
        return np.array(U_save), np.array(t_save)
