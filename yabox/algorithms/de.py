# -*- coding: utf-8 -*-
from .base import *
from collections import deque

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
        self.population = self.init()
        self.fitness = self.evaluate(self.population)
        self.best_fitness = min(self.fitness)
        self.best_idx = np.argmin(self.fitness)
        self.f, self.cr = self.calculate_params()
        # Last mutant created
        self.mutant = None
        # Index of the last target vector used for mutation/crossover
        self.idx_target = 0
        # Current iteration of the algorithm (it is incremented
        # only when all vectors of the population are processed)
        self.iteration = 0
    
    def calculate_params(self):
        return dither(self.mutation_bounds, self.crossover_bounds)

    def create_mutant(self, i):
        # Simple self-adaptive strategy, where the F and CR control
        # parameters are inherited from the base vector.
        if self.adaptive:
            # Use the params of the target vector
            dt = self.denormalize(self.population[i])
            self.f, self.cr = dt[-2:]
        self.mutant = self._mutant(i)
        return self.mutant

    def replace(self, i, mutant):
        mutant_fitness, = self.evaluate(np.asarray([mutant]))
        self.replacement(i, mutant, mutant_fitness)

    def replacement(self, target_idx, mutant, mutant_fitness):
        if mutant_fitness < self.best_fitness:
            self.best_fitness = mutant_fitness
            self.best_idx = target_idx
        if mutant_fitness < self.fitness[target_idx]:
            self.population[target_idx] = mutant
            self.fitness[target_idx] = mutant_fitness

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

    def _mutant(self, target_idx):
        # Create a mutant using a base vector
        trial = self.mutate(target_idx, self.population, self.f)
        # Repair the individual if a gene is out of bounds
        mutant = self.repair(self.crossover(self.population[target_idx], trial, self.cr))
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
        # This is the main DE loop. For each vector (target vector) in the
        # population, a mutant is created by combining different vectors in
        # the population (depending on the strategy selected). If the mutant
        # is better than the target vector, the target vector is replaced.
        while self.iteration < self.maxiters:
            # Compute values for f and cr in each iteration
            self.f, self.cr = self.calculate_params()
            for self.idx_target in range(self.popsize):
                # Create a mutant using a base vector, and the current f and cr values
                mutant = self.create_mutant(self.idx_target)
                # Evaluate and replace if better
                self.replace(self.idx_target, mutant)
                # Yield the current state of the algorithm
            yield self.population, self.fitness, self.best_idx
            self.iteration += 1

    def solve(self, show_progress=False):
        iterator = self.iterator()
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(self.iterator(), total=self.maxiters,
                            desc=f'Optimizing ({self.name})')

        # Check https://docs.python.org/3/library/itertools.html#itertools-recipes
        # feed the entire iterator into a zero-length deque
        deque(iterator, maxlen=0)
              
        if show_progress:
            iterator.n = self.maxiters
            iterator.refresh()
            iterator.close()
        
        best_ind = self.population[self.best_idx].reshape(-1, 1)
        best_ind_denorm = self.denormalize(best_ind)
        return best_ind_denorm, self.best_fitness


class PDE(DE):
    def __init__(self, fobj, bounds, mutation=(0.5, 1.0), crossover=0.7, maxiters=1000,
                 self_adaptive=False, popsize=None, seed=None, processes=None, chunksize=None):
        
        from multiprocessing import Pool
        self.processes = processes
        self.chunksize = chunksize
        self.name = 'Parallel DE'
        self.pool = None
        if processes is None or processes > 0:
            self.pool = Pool(processes=self.processes)
        
        super().__init__(fobj, bounds, mutation, crossover, maxiters, self_adaptive, popsize, seed)

        self.mutants = np.zeros((self.popsize, self.dims))
        
    
    def create_mutant(self, i):
        mutant = super().create_mutant(i)
        # Add to the mutants population for parallel evaluation (later)
        # self.mutants.append(mutant)
        self.mutants[i, :] = mutant

    def replace(self):
        # Evaluate the whole new population (class PDE implements a parallel version of evaluate)
        mutant_fitness = self.evaluate(self.mutants)
        
        # We have to make super().replacement static in other to parallelized it
        # self.pool.starmap(super().replacement, zip(range(self.popsize), self.mutants, mutant_fitness), chunksize=self.chunksize)
        for j, (mutant, fitness) in enumerate(zip(self.mutants, mutant_fitness)):
            super().replacement(j, mutant, fitness)

    def iterator(self):
        # This is the main DE loop. For each vector (target vector) in the
        # population, a mutant is created by combining different vectors in
        # the population (depending on the strategy selected). If the mutant
        # is better than the target vector, the target vector is replaced.
        while self.iteration < self.maxiters:
            # Compute values for f and cr in each iteration
            self.f, self.cr = self.calculate_params()
            for self.idx_target in range(self.popsize):
                # Create a mutant using a base vector, and the current f and cr values
                self.create_mutant(self.idx_target)
            
            # Evaluate and replace if better
            self.replace()
            # Yield the current state of the algorithm
            yield self.population, self.fitness, self.best_idx
            self.iteration += 1

        if self.pool is not None:
            self.pool.terminate()

    def evaluate_denormalized(self, PD):
        if self.pool is not None:
            return list(self.pool.map(self.fobj, PD, chunksize=self.chunksize))
        else:
            return super().evaluate_denormalized(PD)
