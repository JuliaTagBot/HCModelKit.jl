module HCModelKit

using OrderedCollections: OrderedDict
using StaticArrays: @SVector, @SMatrix

import LinearAlgebra: det, dot
import Latexify

export Expression, Constant, Variable, Operation

export @var,
       @unique_var,
       evaluate,
       subs,
       variables,
       differentiate,
       monomials,
       Compiled,
       CompiledSystem,
       CompiledHomotopy,
       compile,
       interpret,
       interpreted,
       System,
       Homotopy,
       evaluate,
       evaluate!,
       evaluate_gradient,
       evaluate_jacobian,
       evaluate_jacobian!,
       jacobian,
       jacobian!,
       dt,
       dt!,
       dt_jacobian,
       dt_jacobian!

include("expression.jl")
include("codegen.jl")

end # module
