# -*- coding: utf-8 -*-
from time import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

# Default configuration
contourParams = dict(
    zdir='z',
    alpha=0.5,
    zorder=1,
    antialiased=True,
    cmap=cm.PuRd_r
)

surfaceParams = dict(
    rstride=1,
    cstride=1,
    linewidth=0.1,
    edgecolors='k',
    alpha=0.5,
    antialiased=True,
    cmap=cm.PuRd_r
)


class BaseProblem:
    def __init__(self, dim=None, bounds=None, default_bounds=(-1, 1), name=None):
        if bounds is None:
            bounds = [default_bounds]
            if dim is not None:
                bounds = [default_bounds] * dim
        self.dimensions = len(bounds)
        self.bounds = bounds
        self.name = name or self.__class__.__name__

    def __call__(self, *args, **kwargs):
        return self.evaluate(*args, **kwargs)

    def evaluate(self, x):
        raise NotImplementedError('Implement the evaluation function')

    def plot2d(self, points=100, figure=None, figsize=(12, 8), contour=True, contour_levels=20,
               imshow_kwds=None, contour_kwds=None):
        if imshow_kwds is None:
            imshow_kwds = dict(cmap=cm.PuRd_r)
        if contour_kwds is None:
            contour_kwds = dict(cmap=cm.PuRd_r)
        xbounds, ybounds = self.bounds[0], self.bounds[1]
        x = np.linspace(min(xbounds), max(xbounds), points)
        y = np.linspace(min(xbounds), max(xbounds), points)
        X, Y = np.meshgrid(x, y)
        Z = self(np.asarray([X, Y]))
        if figure is None:
            fig = plt.figure(figsize=figsize)
        else:
            fig = figure
        ax = fig.gca()
        if contour:
            ax.contourf(X, Y, Z, contour_levels, **contour_kwds)
        else:
            im = ax.imshow(Z, **imshow_kwds)
        if figure is None:
            plt.show()
        return fig, ax

    def plot3d(self, points=100, contour_levels=20, ax3d=None, figsize=(12, 8),
               view_init=None, surface_kwds=None, contour_kwds=None):
        from mpl_toolkits.mplot3d import Axes3D
        contour_settings = dict(contourParams)
        surface_settings = dict(surfaceParams)
        if contour_kwds is not None:
            contour_settings.update(contour_kwds)
        if surface_kwds is not None:
            surface_settings.update(surface_kwds)
        xbounds, ybounds = self.bounds[0], self.bounds[1]
        x = np.linspace(min(xbounds), max(xbounds), points)
        y = np.linspace(min(ybounds), max(ybounds), points)
        X, Y = np.meshgrid(x, y)
        Z = self(np.asarray([X, Y]))
        if ax3d is None:
            fig = plt.figure(figsize=figsize)
            ax = Axes3D(fig)
            if view_init is not None:
                ax.view_init(*view_init)
        else:
            ax = ax3d
        # Make the background transparent
        ax.patch.set_alpha(0.0)
        # Make each axis pane transparent as well
        ax.w_xaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
        ax.w_yaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
        ax.w_zaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
        surf = ax.plot_surface(X, Y, Z, **surface_settings)
        contour_settings['offset'] = np.min(Z)
        cont = ax.contourf(X, Y, Z, contour_levels, **contour_settings)
        if ax3d is None:
            plt.show()
        return ax

    def __repr__(self):
        return '{} {}D'.format(self.name, self.dimensions)


class Slowdown(BaseProblem):
    def __init__(self, problem, us=1000):
        super().__init__(bounds=problem.bounds, name='{} (~{} us)'.format(problem.name, us))
        self.problem = problem
        self.us = us

    def evaluate(self, x):
        start = time() * 1e6
        result = 0
        while time() * 1e6 - start < self.us:
            result = self.problem.evaluate(x)
        return result


class Ackley(BaseProblem):
    def __init__(self, dim=2, bounds=None, default_bounds=(-5, 5), a=20, b=0.2, c=2 * np.pi):
        super().__init__(dim, bounds, default_bounds)
        self.a = a
        self.b = b
        self.c = c

    def evaluate(self, x):
        n = len(x)
        s1 = sum(np.power(x, 2))
        s2 = sum(np.cos(self.c * x))
        return -self.a * np.exp(-self.b * np.sqrt(s1 / n)) - np.exp(s2 / n) + self.a + np.exp(1)


class Rastrigin(BaseProblem):
    def __init__(self, dim=2, bounds=None, default_bounds=(-5.12, 5.12), a=10):
        super().__init__(dim, bounds, default_bounds)
        self.a = a

    def evaluate(self, x):
        d = len(x)
        s = np.power(x, 2) - self.a * np.cos(2 * np.pi * x)
        return self.a * d + sum(s)


class Rosenbrock(BaseProblem):
    def __init__(self, dim=2, bounds=None, default_bounds=(-10, 10), z_shift=0):
        super().__init__(dim, bounds, default_bounds)
        self.z_shift = z_shift

    def evaluate(self, x):
        return sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0) + self.z_shift


class CrossInTray(BaseProblem):
    def __init__(self, bounds=None, default_bounds=(-10, 10)):
        super().__init__(2, bounds, default_bounds)

    def evaluate(self, x):
        x1, x2 = x[0], x[1]
        return -0.0001 * (np.abs(
            np.sin(x1) * np.sin(x2) * np.exp(np.abs(100 - np.sqrt(x1 ** 2 + x2 ** 2) / np.pi))) + 1) ** 0.1


class EggHolder(BaseProblem):
    def __init__(self, bounds=None, default_bounds=(-512, 512)):
        super().__init__(2, bounds, default_bounds)

    def evaluate(self, x):
        x1, x2 = x[0], x[1]
        return -(x2 + 47) * np.sin(np.sqrt(np.abs(x2 + x1 / 2 + 47))) - x1 * np.sin(np.sqrt(np.abs(x1 - (x2 + 47))))


class HolderTable(BaseProblem):
    def __init__(self, bounds=None, default_bounds=(-10, 10)):
        super().__init__(2, bounds, default_bounds)

    def evaluate(self, x):
        x1, x2 = x[0], x[1]
        return -np.abs(np.sin(x1) * np.cos(x2) * np.exp(np.abs(1 - np.sqrt(x1 ** 2 + x2 ** 2) / np.pi)))


class Easom(BaseProblem):
    def __init__(self, bounds=None, default_bounds=(-100, 100)):
        super().__init__(2, bounds, default_bounds)

    def evaluate(self, x):
        x1, x2 = x[0], x[1]
        return -np.cos(x1) * np.cos(x2) * np.exp(-(x1 - np.pi) ** 2 - (x2 - np.pi) ** 2)


class StyblinskiTang(BaseProblem):
    def __init__(self, dim=2, bounds=None, default_bounds=(-5, 5)):
        super().__init__(dim, bounds, default_bounds)

    def evaluate(self, x):
        return sum(x ** 4 - 16 * x ** 2 + 5 * x) / 2


class Michalewicz(BaseProblem):
    def __init__(self, dim=2, bounds=None, default_bounds=(0, np.pi), m=10):
        super().__init__(dim, bounds, default_bounds)
        self.m = m

    def evaluate(self, x):
        c = 0
        for i in range(0, len(x)):
            c += np.sin(x[i]) * np.sin(( (i+1) * x[i]**2)/np.pi) ** (2*self.m)
        return -c


class Schwefel(BaseProblem):
    def __init__(self, dim=2, bounds=None, default_bounds=(-500, 500)):
        super().__init__(dim, bounds, default_bounds)

    def evaluate(self, x):
        d = len(x)
        return 418.9829*d - sum(x*np.sin(np.sqrt(np.abs(x))))


class Levy(BaseProblem):
    def __init__(self, dim=2, bounds=None, default_bounds=(-10, 10)):
        super().__init__(dim, bounds, default_bounds)

    def evaluate(self, x):
        w = 1 + (x - 1) / 4
        wp = w[:-1]
        wd = w[-1]
        a = np.sin(np.pi * w[0]) ** 2
        b = sum((wp - 1) ** 2 * (1 + 10 * np.sin(np.pi * wp + 1) ** 2))
        c = (wd - 1) ** 2 * (1 + np.sin(2 * np.pi * wd) ** 2)
        return a + b + c


class DixonPrice(BaseProblem):
    def __init__(self, dim=2, bounds=None, default_bounds=(-10, 10)):
        super().__init__(dim, bounds, default_bounds)

    def evaluate(self, x):
        c = 0
        for i in range(1, len(x)):
            c += i * (2 * x[i] ** 2 - x[i-1]) ** 2
        return (x[0] - 1) ** 2 + c


class Griewank(BaseProblem):
    def __init__(self, dim=2, bounds=None, default_bounds=(-600, 600)):
        super().__init__(dim, bounds, default_bounds)

    def evaluate(self, x):
        a = sum(x ** 2 / 4000)
        b = 1
        for i in range(len(x)):
            b *= np.cos(x[i] / np.sqrt(i+1))
        return a - b + 1
