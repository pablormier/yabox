def define_problems(function, start=1, end=7):
    return [function(dim = 2 ** i) for i in range(start, end)]


def benchmark(problem, method, popsize, stop_after, maxiters, mutation, recombination):
    times = []
    fitness = []
    start = time()
    if method == 'scipy_de':
        def callback(xk, convergence):
            curr = time()-start
            fitness.append(problem(xk))
            times.append(curr)
            if curr > stop_after:
                return True
            return False
        multiplier = int(round(popsize / len(problem.bounds)))
        if multiplier * len(problem.bounds) != popsize:
            raise ValueError('Invalid value for popsize for the number of parameters.')
        result = SDE(problem, problem.bounds, strategy='rand1bin', popsize=multiplier, maxiter=maxiters,
                     mutation=mutation, recombination=recombination, polish=False, tol=-1, init='random', 
                     callback=callback)
    else:
        if method == 'yabox_de':
            it = DE(problem, problem.bounds, popsize=popsize, maxiters=maxiters, mutation=mutation,
                    crossover=recombination, self_adaptive=False).iterator()
        elif method == 'yabox_pde':
            it = PDE(problem, problem.bounds, popsize=popsize, maxiters=maxiters, mutation=mutation,
                     crossover=recombination, self_adaptive=False).iterator()
        i = 0
        for status in it:
            curr = time()-start
            times.append(curr)
            fitness.append(status.best_fitness)
            i += 1
            if i > maxiters or curr > stop_after:
                break
            
    return times, fitness

def run(problem, method, runs, popsize, stop_after, maxiters, mutation, recombination):
    ts, fs = [], []
    for i in range(runs):
        t, f = benchmark(problem, method, popsize, stop_after, maxiters, mutation, recombination)
        ts.append(t)
        fs.append(f)
    return ts, fs

def run_benchmark(problems, methods, runs, popsize, stop_after, maxiters, mutation, recombination):
    results = []
    for problem in problems:
        method_result = []
        for method in methods:
            print('Running', method, 'on', problem, '({} runs, {} seconds per run)'.format(runs, stop_after))
            t, f = run(problem, method, runs, popsize, stop_after, maxiters, mutation, recombination)
            method_result.append((str(problem), method, t, f))
        results.append(method_result)
    print('Benchmark completed')
    return results

def run_default(problems):
    return run_benchmark(problems, methods, runs, popsize, stop_after, maxiters, mutation, recombination)

def trunc(array):
    l = len(array[0])
    for a in array:
        l = min(l, len(a))
    return np.asarray([a[:l] for a in array])

def plot_results(data, rows=2, columns=3, use_time=True):
    plt.figure(figsize=(16, 10))
    value = 0
    for problem in data:
        value += 1
        plt.subplot(rows, columns, value)
        if use_time:
            plt.xlabel('time (seconds)')
        else:
            plt.xlabel('iteration')
        plt.ylabel('$f(x)$')
        for method in problem:
            problem, algorithm, t, f = method
            t_avg = np.average(trunc(t), axis=0)
            f_avg = np.average(trunc(f), axis=0)
            if use_time:
                plt.plot(t_avg, f_avg, label=algorithm)
            else:
                plt.plot(f_avg, label=algorithm)
            plt.title(problem)
        plt.legend()
    plt.show()   
    
def plot_time_per_iteration(data):
    plt.figure(figsize=(16, 5))
    # Calculate the average time per iteration for each problem
    method_dict = {}
    problems = []
    for problem in data:
        for method in problem:
            problem, algorithm, t, f = method
            if problem not in problems:
                problems.append(problem)
            t_avg = np.average(np.diff(trunc(t)), axis=0)
            # Compute the average per iteration
            iter_avg = np.average(t_avg)
            l = method_dict.get(algorithm, [])
            l.append(iter_avg)
            method_dict[algorithm] = l
    for alg, values in method_dict.items():
        plt.plot(method_dict[alg], '-*', label=alg)
        plt.xticks(range(len(problems)), problems)
        plt.ylabel('time per iteration (seconds)')
    plt.legend()
    plt.show()
    
def test(function):
    # Create different ackley function (2D, 4D, 8D, ... up to 64D)
    problems = define_problems(function)
    # Run each algorithm X times (default 3 times) for each dimension, and average the final result
    return run_default(problems) 
