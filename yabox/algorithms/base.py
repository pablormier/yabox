# -*- coding: utf-8 -*-
import numpy as np


def binomial_crossover(target, mutant, cr):
    n = len(target)
    p = np.random.rand(n) < cr
    if not np.any(p):
        p[np.random.randint(0, n)] = True
    return np.where(p, mutant, target)


def random_sample(population, exclude, size=3):
    # Optimized version using numpy
    idxs = list(range(population.shape[0]))
    idxs.remove(exclude)
    sample = np.random.choice(idxs, size=size, replace=False)
    return population[sample]


def rand1(target_idx, population, f):
    a, b, c = random_sample(population, target_idx)
    return a + f * (b - c)


def denormalize(min_, diff, matrix):
    return min_ + matrix * diff


def random_repair(x):
    # Detect the positions where the params is not valid
    loc = np.logical_or(x < 0, x > 1)
    # Count the number of invalid params
    count = np.sum(loc)
    # Replace each position where a True appears by a new random number in [0-1]
    if count > 0:
        np.place(x, loc, np.random.rand(count))
    return x


def bound_repair(x):
    return np.clip(x, 0, 1)


def random_init(popsize, dimensions):
    return np.random.rand(popsize, dimensions)


def dither_from_interval(interval):
    low, up = min(interval), max(interval)
    if low == up:
        return low
    return np.random.uniform(low, up)


def dither(*intervals):
    return [dither_from_interval(interval) for interval in intervals]
