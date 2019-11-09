module HCModelKit

using OrderedCollections: OrderedDict
using StaticArrays: @SVector

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
       CompiledOperation,
       CompiledOperationSystem,
       CompiledOperationHomotopy,
       compile,
       evaluate,
       evaluate!,
       evaluate_gradient,
       evaluate_jacobian!,
       jacobian,
       jacobian!

 include("expression.jl")
include("codegen.jl")

end # module
