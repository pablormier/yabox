# -*- coding: utf-8 -*-
from yabox import DE, PDE
from yabox.problems import BaseProblem, Ackley


class Fun(BaseProblem):
    def evaluate(self, x):
        return sum(x ** 2)


def test_de_simple_fun():
    x, f = DE(lambda x: sum(x ** 2), [(-10, 10)]).solve()
    assert f == 0


def test_pde_simple_fun():
    x, f = PDE(Fun(), [(-10, 10)]).solve()
    assert f == 0


def test_de_adaptive():
    x, f = DE(lambda x: sum(x ** 2), [(-10, 10)], self_adaptive=True).solve()
    # Check variance and average values of c and f
    pass


def test_de_ackley_fun():
    problem = Ackley()
    x, f = DE(problem, problem.bounds).solve()
    assert f < 1e-8

