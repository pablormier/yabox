# -*- coding: utf-8 -*-
import numpy as np


class DEIterator:
    def __init__(self, de):
        self.de = de
        self.population = de.init()
        self.fitness = de.evaluate(self.population)
        self.best_fitness = min(self.fitness)
        self.best_idx = np.argmin(self.fitness)
        self.f, self.cr = de.dither(de.mutation_bounds, de.crossover_bounds)
        self.iteration = 0

    def __iter__(self):
        de = self.de
        while True:
            # Compute values for f and cr in each iteration
            self.f, self.cr = self.calculate_params()
            for i in range(de.popsize):
                # Create a mutant using a base vector, and the current f and cr values
                mutant = self.create_mutant(i)
                # Evaluate and replace if better
                self.replace(i, mutant)
            self.iteration += 1
            yield self

    def calculate_params(self):
        return self.de.dither(self.de.mutation_bounds, self.de.crossover_bounds)

    def create_mutant(self, i):
        return self.de.mutant(i, self.population, self.f, self.cr)

    def replace(self, i, mutant):
        mutant_fitness, = self.de.evaluate([mutant])
        return self.replacement(i, mutant, mutant_fitness)

    def replacement(self, target_idx, mutant, mutant_fitness):
        if mutant_fitness < self.best_fitness:
            self.best_fitness = mutant_fitness
            self.best_idx = target_idx
        if mutant_fitness < self.fitness[target_idx]:
            self.population[target_idx] = mutant
            self.fitness[target_idx] = mutant_fitness
            return True
        return False


class PDEIterator(DEIterator):
    def __init__(self, de):
        super().__init__(de)
        self.mutants = []

    def create_mutant(self, i):
        mutant = super().create_mutant(i)
        # Add to the mutants population for parallel evaluation (later)
        self.mutants.append(mutant)
        return mutant

    def replace(self, i, mutant):
        # Do not analyze after having the whole population (wait until the last individual)
        if i == self.de.popsize - 1:
            # Evaluate the whole new population (class PDE implements a parallel version of evaluate)
            mutant_fitness = self.de.evaluate(self.mutants)
            for j in range(self.de.popsize):
                super().replacement(j, self.mutants[j], mutant_fitness[j])
            # Clear the mutants population for the next iteration
            self.mutants = []


class DE:
    def __init__(self, fobj, bounds, mutation=(0.5, 1.0), crossover=0.7, maxiters=1000,
                 self_adaptive=False, popsize=None, seed=None):
        self.adaptive = self_adaptive
        # Indicates the number of extra parameters in an individual that are not used for evaluating
        # If extra_params = d, discards the last d elements from an individual prior to evaluation.
        self.extra_params = 0
        # Convert crossover param to an interval, as in mutation. If min/max values in the interval are
        # different, a dither mechanism is used for crossover (although this is not recommended, but still supported)
        # TODO: Clean duplicate code
        if hasattr(crossover, '__len__'):
            self.crossover_bounds = crossover
        else:
            self.crossover_bounds = (crossover, crossover)
        if hasattr(mutation, '__len__'):
            self.mutation_bounds = mutation
        else:
            self.mutation_bounds = (mutation, mutation)
        # If self-adaptive, include mutation and crossover as two new variables
        bnd = list(bounds)
        if self_adaptive:
            bnd.append(self.mutation_bounds)
            bnd.append(self.crossover_bounds)
            self.extra_params = 2
        B = np.asarray(bnd).T
        self._MIN = B[0]
        self._MAX = B[1]
        self._DIFF = np.fabs(self._MAX - self._MIN)
        self.dims = len(bnd)
        self.fobj = fobj
        self.maxiters = maxiters
        if popsize is None:
            # By default it uses 15 individuals for each dimension (as in the scipy's version)
            self.popsize = self.dims * 15
        else:
            self.popsize = popsize
        # Initialize random state
        if seed is None:
            self.rnd = np.random.RandomState()
        elif isinstance(seed, np.random.RandomState):
            self.rnd = seed
        else:
            self.rnd = np.random.RandomState(seed)
        # Default functions
        self.crossover = getattr(self, '_binomial_crossover')
        self.mutate = getattr(self, '_rand1')
        self.repair = getattr(self, '_random_repair')
        self.init = getattr(self, '_random_init')
        self.name = 'DE'

    def sample(self, exclude_index, P, size):
        idx = [i for i in range(len(P)) if i != exclude_index]
        self.rnd.shuffle(idx)
        return P[idx[:size]]

    def mutant(self, target_idx, population, f, cr):
        # Create a mutant using a base vector
        trial = self.mutate(target_idx, population, f)
        # Repair the individual if a gene is out of bounds
        mutant = self.repair(self.crossover(population[target_idx], trial, cr))
        return mutant

    def evaluate(self, P):
        # Denormalize population matrix to obtain the scaled parameters
        PD = self.denormalize(P)
        if self.extra_params > 0:
            PD = PD[:, :-self.extra_params]
        return [self.fobj(ind) for ind in PD]

    def _rand1(self, target, P, mutation_factor):
        a, b, c = self.sample(target, P, 3)
        return a + mutation_factor * (b - c)

    def dither_from_interval(self, interval):
        low, up = min(interval), max(interval)
        if low == up:
            return low
        return self.rnd.uniform(low, up)

    def dither(self, *intervals):
        return [self.dither_from_interval(interval) for interval in intervals]

    def iterator(self):
        return DEIterator(self)

    def solve(self, show_progress=False):
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(self.iterator(), total=self.maxiters, desc='Optimizing ({0})'.format(self.name))
        else:
            iterator = self.iterator()
        step = 0
        for status in iterator:
            idx = status.best_idx
            P = status.population
            fitness = status.fitness
            step += 1
            if step > self.maxiters:
                if show_progress:
                    iterator.n = self.maxiters
                    iterator.refresh()
                    iterator.close()
                return self.denormalize(P[idx]), fitness[idx]

    def denormalize(self, P):
        return self._MIN + P * self._DIFF

    def _binomial_crossover(self, target, mutant, probability):
        n = len(target)
        p = self.rnd.rand(n) < probability
        if not np.any(p):
            p[self.rnd.randint(0, n)] = True
        return np.where(p, mutant, target)

    def _random_repair(self, x):
        # Detect the positions where the para is not valid
        loc = np.logical_or(x < 0, x > 1)
        # Count the number of invalid genes
        count = np.sum(loc)
        # Replace each position where a True appears by a new random number
        if count > 0:
            np.place(x, loc, self.rnd.rand(count))
        return x

    def _bound_repair(self, x):
        return np.clip(x, 0, 1)

    def _lhs_init(self):
        pass

    def _random_init(self):
        return self.rnd.rand(self.popsize, self.dims)


class PDE(DE):
    def __init__(self, fobj, bounds, mutation=(0.5, 1.0), crossover=0.7, maxiters=1000,
                 self_adaptive=False, popsize=None, seed=None, processes=None, chunksize=None):
        super().__init__(fobj, bounds, mutation, crossover, maxiters, self_adaptive, popsize, seed)
        from multiprocessing import Pool
        self.processes = processes
        self.chunksize = chunksize
        self.name = 'Parallel DE'
        self.pool = Pool(processes=self.processes)

    def iterator(self):
            it = PDEIterator(self)
            try:
                for data in it:
                    yield data
            finally:
                self.pool.terminate()

    def evaluate(self, P):
        PD = self.denormalize(P)
        extra = self.extra_params
        if extra > 0:
            PD = PD[:, :-extra]
        return list(self.pool.map(self.fobj, PD, chunksize=self.chunksize))
