# -*- coding: utf-8 -*-
from yabox import DE, PDE
from yabox.problems import problem


def test_de_simple_fun():
    x, f = DE(lambda x: sum(x ** 2), [(-10, 10)]).solve()
    assert f == 0

