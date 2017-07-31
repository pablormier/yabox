# -*- coding: utf-8 -*-
import numpy as np


class DE:
    def __init__(self, fobj, bounds, mutation=(0.5, 1.0), crossover=0.7, maxiters=1000,
                 popsize=None, seed=None):
        B = np.asarray(bounds).T
        self._MIN = B[0]
        self._MAX = B[1]
        self._DIFF = np.fabs(self._MAX - self._MIN)
        self.dims = len(bounds)
        self.crs = crossover
        self.fobj = fobj
        self.maxiters = maxiters
        self.mutation = mutation
        if popsize is None:
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

    def _sample(self, exclude_index, P, size):
        idx = [i for i in range(len(P)) if i != exclude_index]
        selected = np.random.choice(idx, size, replace=False)
        return P[selected]

    def _eval_population(self, P):
        # Denormalize population matrix to obtain the scaled parameters
        PD = self._denorm(P)
        return [self.fobj(ind) for ind in PD]

    def _eval(self, ind):
        return self.fobj(self._denorm(ind))

    def _rand1(self, target, P, mutation_factor):
        a, b, c = self._sample(target, P, 3)
        return a + mutation_factor * (b - c)

    def __iter__(self):
        # Initialize a random population of solutions
        P = self.init()
        fitness = self._eval_population(P)
        best_fitness = min(fitness)
        best_idx = np.argmin(fitness)
        while True:
            # Apply dithering on each iteration
            mutation_factor = np.random.uniform(min(self.mutation), max(self.mutation))
            for i in range(self.popsize):
                target = P[i]
                # Create a mutant using a base vector
                mutant = self.mutate(i, P, mutation_factor)
                # Repair the individual if a gene is out of bounds
                z = self.repair(self.crossover(target, mutant, self.crs))
                new_fitness = self._eval(z)
                if new_fitness < best_fitness:
                    best_fitness = new_fitness
                    best_idx = i
                if new_fitness < fitness[i]:
                    # The new solution is better, replace
                    P[i] = z
                    fitness[i] = new_fitness
            yield P, fitness, best_idx

    def iterator(self, denormalize=True):
        for P, fitness, idx in self:
            pop = self._denorm(P) if denormalize else P
            yield pop, fitness, idx

    def solve(self, show_progress=False):
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(self, total=self.maxiters, desc='Optimizing (DE)')
        else:
            iterator = self
        step = 0
        for P, fitness, idx in iterator:
            step += 1
            if step > self.maxiters:
                if show_progress:
                    iterator.n = self.maxiters
                    iterator.refresh()
                    iterator.close()
                return self._denorm(P[idx]), fitness[idx]

    def _denorm(self, P):
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
    def __init__(self, fobj, bounds, mutation=(0.5, 1.0), crossover=0.7,
                 maxiters=1000, popsize=None, seed=None, processes=None, chunksize=None):
        super().__init__(fobj, bounds, mutation, crossover, maxiters, popsize, seed)
        from multiprocessing import Pool
        self.pool = Pool(processes=processes)
        self.chunksize = chunksize

    def __iter__(self):
        # Initialize a random population of solutions
        P = self.init()
        fitness = self._eval_population(P)
        best_fitness = min(fitness)
        best_idx = np.argmin(fitness)
        while True:
            # Apply dithering on each iteration
            mutation_factor = np.random.uniform(min(self.mutation), max(self.mutation))
            # Create mutants
            mutants = []
            for i in range(self.popsize):
                target = P[i]
                # Create a mutant using a base vector
                mutant = self.mutate(i, P, mutation_factor)
                # Repair the individual if a gene is out of bounds
                z = self.repair(self.crossover(target, mutant, self.crs))
                mutants.append(z)
            # Evaluate in parallel
            new_fitness = self._eval_population(mutants)
            # Replace if better
            for i in range(self.popsize):
                if new_fitness[i] < fitness[i]:
                    P[i] = mutants[i]
                    fitness[i] = new_fitness[i]
                if new_fitness[i] < best_fitness:
                    best_fitness = new_fitness[i]
                    best_idx = i
            yield P, fitness, best_idx

    def _eval_population(self, P):
        # Denormalize population matrix to obtain the scaled parameters
        PD = self._denorm(P)
        return list(self.pool.map(self.fobj, PD, chunksize=self.chunksize))
