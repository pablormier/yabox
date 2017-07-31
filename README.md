<img src='../master/docs/images/yabox.png?raw=true' width=250 align=right />

# Yabox

_Yet another black-box optimization library for Python_

## Description

Yabox is a small library for black-box (derivative free) optimization of functions. The library includes different stochastic algorithms for minimizing a function f(X), where X = {x1, ..., xN}. The function f(X) does not need to have an analytical form.
The current version of the library includes the Differential Evolution algorithm and a modified version for parallel evaluation.

Example of minimization of the Ackley function (using Yabox and Differential Evolution):

![Ackley Function](../master/notebooks/img/ackley.gif?raw=true)


