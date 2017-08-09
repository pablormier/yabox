# -*- coding: utf-8 -*-
import numpy as np


class DEIterator:
    def __init__(self, de):
        self.de = de

    def __iter__(self):
        de = self.de
        # Initialize a random population of solutions
        P = de.init()
        fitness = de.evaluate(P)
        best_fitness = min(fitness)
        best_idx = np.argmin(fitness)
        while True:
            # Apply dithering on each iteration
            if not de.adaptive:
                f = de.dither(de.fmut)
                c = de.dither(de.c)

            for i in range(de.popsize):
                target = P[i]
                if de.adaptive:
                    # Denormalize and read the values for the mutation and crossover
                    dt = de.denormalize(target)
                    f, c = dt[-2:]
                # Create a mutant using a base vector
                mutant = de.mutate(i, P, f)
                # Repair the individual if a gene is out of bounds
                z = de.repair(de.crossover(target, mutant, c))
                new_fitness, = de.evaluate([z])
                if new_fitness < best_fitness:
                    best_fitness = new_fitness
                    best_idx = i
                if new_fitness < fitness[i]:
                    # The new solution is better, replace
                    P[i] = z
                    fitness[i] = new_fitness
            yield P, fitness, best_idx


class PDEIterator(DEIterator):
    def __init__(self, de):
        super().__init__(de)

    def __iter__(self):
        de = self.de
        # Initialize a random population of solutions
        P = de.init()
        fitness = de.evaluate(P)
        best_fitness = min(fitness)
        best_idx = np.argmin(fitness)
        while True:
            # Apply dithering on each iteration
            if not de.adaptive:
                f = de.dither(de.fmut)
                c = de.dither(de.c)

            mutants = []
            for i in range(de.popsize):
                target = P[i]
                if de.adaptive:
                    # Denormalize and read the values for the mutation and crossover
                    dt = de.denormalize(target)
                    f, c = dt[-2:]
                # Create a mutant using a base vector
                mutant = de.mutate(i, P, f)
                # Repair the individual if a gene is out of bounds
                z = de.repair(de.crossover(target, mutant, c))
                mutants.append(z)
            # (Parallel) evaluation of the new candidates
            new_fitness = de.evaluate(mutants)
            # Replace if better
            for i in range(de.popsize):
                if new_fitness[i] < fitness[i]:
                    P[i] = mutants[i]
                    fitness[i] = new_fitness[i]
                if new_fitness[i] < best_fitness:
                    best_fitness = new_fitness[i]
                    best_idx = i
            yield P, fitness, best_idx


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
            self.c = crossover
        else:
            self.c = (crossover, crossover)
        if hasattr(mutation, '__len__'):
            self.fmut = mutation
        else:
            self.fmut = (mutation, mutation)
        # If self-adaptive, include mutation and crossover as two new variables
        bnd = list(bounds)
        if self_adaptive:
            bnd.append(self.fmut)
            bnd.append(self.c)
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
        selected = self.rnd.choice(idx, size, replace=False)
        return P[selected]

    def evaluate(self, P):
        # Denormalize population matrix to obtain the scaled parameters
        PD = self.denormalize(P)
        if self.extra_params > 0:
            PD = PD[:, :-self.extra_params]
        return [self.fobj(ind) for ind in PD]

    def _rand1(self, target, P, mutation_factor):
        a, b, c = self.sample(target, P, 3)
        return a + mutation_factor * (b - c)

    def dither(self, interval):
        low, up = min(interval), max(interval)
        if low == up:
            return low
        return self.rnd.uniform(low, up)

    def iterator(self):
        return DEIterator(self)

    def solve(self, show_progress=False):
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(self.iterator(), total=self.maxiters, desc='Optimizing ({0})'.format(self.name))
        else:
            iterator = self.iterator()
        step = 0
        for P, fitness, idx in iterator:
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
        z = np.copy(target)
        n = len(z)
        k = self.rnd.randint(0, n)
        for i in range(n):
            if self.rnd.uniform() <= probability or i == k:
                # transfer gene from mutant
                z[i] = mutant[i]
        return z

    def _random_repair(self, x):
        for i in range(len(x)):
            if x[i] < 0 or x[i] > 1:
                x[i] = self.rnd.uniform()
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
        self.processes = processes
        self.chunksize = chunksize
        self.name = 'Parallel DE'
        self.pool = None

    def iterator(self):
            it = PDEIterator(self)
            try:
                from multiprocessing import Pool
                self.pool = Pool(processes=self.processes)
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
