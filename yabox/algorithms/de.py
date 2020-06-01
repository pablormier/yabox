# -*- coding: utf-8 -*-
from .base import *


class DEIterator:
    def __init__(self, de):
        self.de = de
        self.population = de.init()
        self.fitness = de.evaluate(self.population)
        self.best_fitness = min(self.fitness)
        self.best_idx = np.argmin(self.fitness)
        # F and CR control parameters
        self.f, self.cr = dither(de.mutation_bounds, de.crossover_bounds)
        # Last mutant created
        self.mutant = None
        # Index of the last target vector used for mutation/crossover
        self.idx_target = 0
        # Current iteration of the algorithm (it is incremented
        # only when all vectors of the population are processed)
        self.iteration = 0

    def __iter__(self):
        de = self.de
        # This is the main DE loop. For each vector (target vector) in the
        # population, a mutant is created by combining different vectors in
        # the population (depending on the strategy selected). If the mutant
        # is better than the target vector, the target vector is replaced.
        while self.iteration <= self.de.maxiters:
            # Compute values for f and cr in each iteration
            self.f, self.cr = self.calculate_params()
            for self.idx_target in range(de.popsize):
                # Create a mutant using a base vector, and the current f and cr values
                mutant = self.create_mutant(self.idx_target)
                # Evaluate and replace if better
                self.replace(self.idx_target, mutant)
                # Yield the current state of the algorithm
                yield self
            self.iteration += 1

    def calculate_params(self):
        return dither(self.de.mutation_bounds, self.de.crossover_bounds)

    def create_mutant(self, i):
        # Simple self-adaptive strategy, where the F and CR control
        # parameters are inherited from the base vector.
        if self.de.adaptive:
            # Use the params of the target vector
            dt = self.de.denormalize(self.population[i])
            self.f, self.cr = dt[-2:]
        self.mutant = self.de.mutant(i, self.population, self.f, self.cr)
        return self.mutant

    def replace(self, i, mutant):
        mutant_fitness, = self.de.evaluate(np.asarray([mutant]))
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
        self.mutants = np.zeros((de.popsize, de.dims))

    def create_mutant(self, i):
        mutant = super().create_mutant(i)
        # Add to the mutants population for parallel evaluation (later)
        # self.mutants.append(mutant)
        self.mutants[i, :] = mutant
        return mutant

    def replace(self, i, mutant):
        # Do not analyze after having the whole population (wait until the last individual)
        if i == self.de.popsize - 1:
            # Evaluate the whole new population (class PDE implements a parallel version of evaluate)
            mutant_fitness = self.de.evaluate(self.mutants)
            for j in range(self.de.popsize):
                super().replacement(j, self.mutants[j], mutant_fitness[j])


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

        self.crossover_bounds = crossover
        self.mutation_bounds = mutation

        if getattr(crossover, '__len__', None) is None:
            self.crossover_bounds = [crossover, crossover]

        if getattr(mutation, '__len__', None) is None:
            self.mutation_bounds = [mutation, mutation]

        # If self-adaptive, include mutation and crossover as two new variables
        bnd = list(bounds)
        if self_adaptive:
            bnd.append(self.mutation_bounds)
            bnd.append(self.crossover_bounds)
            self.extra_params = 2
        self._MIN, self._MAX = np.asarray(bnd, dtype='f8').T
        self._DIFF = np.fabs(self._MAX - self._MIN)
        self.dims = len(bnd)
        self.fobj = fobj
        self.maxiters = maxiters
        if popsize is None:
            self.popsize = self.dims * 5
        else:
            self.popsize = popsize
        self.initialize_random_state(seed)
        self.name = 'DE'

    @staticmethod
    def initialize_random_state(seed):
        np.random.seed(seed)

    @staticmethod
    def crossover(target, mutant, probability):
        return binomial_crossover(target, mutant, probability)

    @staticmethod
    def mutate(target_idx, population, f):
        return rand1(target_idx, population, f)

    @staticmethod
    def repair(x):
        return random_repair(x)

    def init(self):
        return random_init(self.popsize, self.dims)

    def denormalize(self, population):
        return denormalize(self._MIN, self._DIFF, population)

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
        return self.evaluate_denormalized(PD)

    def evaluate_denormalized(self, PD):
        return [self.fobj(ind) for ind in PD]

    def iterator(self):
        return iter(DEIterator(self))

    def geniterator(self):
        it = self.iterator()
        iteration = 0
        for step in it:
            if step.iteration != iteration:
                iteration = step.iteration
                yield step

    def solve(self, show_progress=False):
        if show_progress:
            from tqdm.auto import tqdm
            iterator = tqdm(self.geniterator(), total=self.maxiters, desc='Optimizing ({0})'.format(self.name))
        else:
            iterator = self.geniterator()
        for step in iterator:
            idx = step.best_idx
            P = step.population
            fitness = step.fitness
        return self.denormalize(P[idx].reshape(1, -1)), fitness[idx]


class PDE(DE):
    def __init__(self, fobj, bounds, mutation=(0.5, 1.0), crossover=0.7, maxiters=1000,
                 self_adaptive=False, popsize=None, seed=None, processes=None, chunksize=None):
        super().__init__(fobj, bounds, mutation, crossover, maxiters, self_adaptive, popsize, seed)
        from multiprocessing import Pool
        self.processes = processes
        self.chunksize = chunksize
        self.name = 'Parallel DE'
        self.pool = None
        if processes is None or processes > 0:
            self.pool = Pool(processes=self.processes)

    def iterator(self):
        it = PDEIterator(self)
        try:
            for data in it:
                yield data
        finally:
            if self.pool is not None:
                self.pool.terminate()

    def evaluate_denormalized(self, PD):
        if self.pool is not None:
            return list(self.pool.map(self.fobj, PD, chunksize=self.chunksize))
        else:
            return super().evaluate_denormalized(PD)
