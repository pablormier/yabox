# -*- coding: utf-8 -*-
from time import time
import types

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

# Default configuration
contour_default_settings = dict(
    zdir='z',
    alpha=0.5,
    zorder=1,
    antialiased=True,
    cmap=cm.PuRd_r
)

surface_default_settings = dict(
    rstride=1,
    cstride=1,
    linewidth=0.1,
    edgecolors='k',
    alpha=0.5,
    antialiased=True,
    cmap=cm.PuRd_r
)


class BaseProblem:
    def __init__(self, bounds):
        self.dimensions = len(bounds)
        self.bounds = bounds

    def __call__(self, *args, **kwargs):
        return self.evaluate(*args, **kwargs)

    def evaluate(self, x):
        raise NotImplementedError('Implement the evaluation function')

    def plot2d(self, points=100, figure=None, figsize=(12, 8), imshow_kwds=None):
        if imshow_kwds is None:
            imshow_kwds = dict(cmap=cm.PuRd_r)
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
        im = ax.imshow(Z, **imshow_kwds)
        if figure is None:
            plt.show()
        return fig, ax

    def plot3d(self, points=100, contour_levels=20, ax3d=None, figsize=(12, 8),
               view_init=None, surface_kwds=None, contour_kwds=None):
        contour_settings = dict(contour_default_settings)
        surface_settings = dict(surface_default_settings)
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
            ax = fig.gca(projection='3d')
            if view_init is not None:
                ax.view_init(*view_init)
        else:
            ax = ax3d
        surf = ax.plot_surface(X, Y, Z, **surface_settings)
        contour_settings['offset'] = np.min(Z)
        cont = ax.contourf(X, Y, Z, contour_levels, **contour_settings)
        if ax3d is None:
            plt.show()
        return ax


class Slowdown(BaseProblem):
    def __init__(self, problem, us=1000):
        super().__init__(problem.bounds)
        self.problem = problem
        self.us = us

    def evaluate(self, x):
        start = time() * 1e6
        result = 0
        while time() * 1e6 - start < self.us:
            result = self.problem.evaluate(x)
        return result


class Ackley(BaseProblem):
    def __init__(self, bounds=[(-5, 5)] * 2, a=20, b=0.2, c=2 * np.pi):
        super().__init__(bounds)
        self.a = a
        self.b = b
        self.c = c

    def evaluate(self, x):
        n = len(x)
        s1 = sum(np.power(x, 2))
        s2 = sum(np.cos(self.c * x))
        return -self.a * np.exp(-self.b * np.sqrt(s1 / n)) - np.exp(s2 / n) + self.a + np.exp(1)


class Rastrigin(BaseProblem):
    def __init__(self, bounds=[(-5.12, 5.12)] * 2, a=10):
        super().__init__(bounds)
        self.a = a

    def evaluate(self, x):
        d = len(x)
        s = np.power(x, 2) - self.a * np.cos(2 * np.pi * x)
        return self.a * d + sum(s)


class Rosenbrock(BaseProblem):
    def __init__(self, bounds=[(-10, 10)] * 2, z_shift=0):
        super().__init__(bounds)
        self.z_shift = z_shift

    def evaluate(self, x):
        return sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0) + self.z_shift


class CrossInTray(BaseProblem):
    def __init__(self, bounds=[(-10, 10)] * 2):
        super().__init__(bounds)

    def evaluate(self, x):
        x1, x2 = x[0], x[1]
        return -0.0001 * (np.abs(
            np.sin(x1) * np.sin(x2) * np.exp(np.abs(100 - np.sqrt(x1 ** 2 + x2 ** 2) / np.pi))) + 1) ** 0.1


class EggHolder(BaseProblem):
    def __init__(self, bounds=[(-512, 512)] * 2):
        super().__init__(bounds)

    def evaluate(self, x):
        x1, x2 = x[0], x[1]
        return -(x2 + 47) * np.sin(np.sqrt(np.abs(x2 + x1 / 2 + 47))) - x1 * np.sin(np.sqrt(np.abs(x1 - (x2 + 47))))


class HolderTable(BaseProblem):
    def __init__(self, bounds=[(-10, 10)] * 2):
        super().__init__(bounds)

    def evaluate(self, x):
        x1, x2 = x[0], x[1]
        return -np.abs(np.sin(x1) * np.cos(x2) * np.exp(np.abs(1 - np.sqrt(x1 ** 2 + x2 ** 2) / np.pi)))


class Easom(BaseProblem):
    def __init__(self, bounds=[(-100, 100)] * 2):
        super().__init__(bounds)

    def evaluate(self, x):
        x1, x2 = x[0], x[1]
        return -np.cos(x1) * np.cos(x2) * np.exp(-(x1 - np.pi) ** 2 - (x2 - np.pi) ** 2)


def problem(f, bounds):
    p = BaseProblem(bounds)
    p.evaluate.__code__ = f.__code__
    return p
