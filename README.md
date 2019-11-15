# HCModelKit

[![Build Status](https://travis-ci.com/JuliaHomotopyContinuation/HCModelKit.jl.svg?branch=master)](https://travis-ci.com/JuliaHomotopyContinuation/HCModelKit.jl)
[![Codecov](https://codecov.io/gh/JuliaHomotopyContinuation/HCModelKit.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/JuliaHomotopyContinuation/HCModelKit.jl)


```julia
using HCModelKit

@var x y z
f = [(0.3 * x^2 + 0.5z + 0.3x + 1.2 * y^2 - 1.1)^2 +
    (0.7 * (y - 0.5x)^2 + y + 1.2 * z^2 - 1)^2 - 0.3]

vars = variables(f)
n, m = length(vars), length(f)

@unique_var y[1:n] v[1:m] w[1:m]

J = differentiate(f, vars)
f′ = subs(f, vars => y)
J′ = subs(J, vars => y)
Nx = (vars - y) - J' * v
Ny = (vars - y) - J′' * w

F = compile(System([f; f′; Nx; Ny], [vars; y; v; w]))
```
