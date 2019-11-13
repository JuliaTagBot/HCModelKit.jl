abstract type Expression end

###############
## VARIABLES ##
###############

struct Variable <: Expression
    name::Symbol
end
Variable(name, indices...) = Variable("$(name)$(join(map_subscripts.(indices), "₋"))")
Variable(s::AbstractString) = Variable(Symbol(s))

const INDEX_MAP = Dict{Char,Char}(
    '0' => '₀',
    '1' => '₁',
    '2' => '₂',
    '3' => '₃',
    '4' => '₄',
    '5' => '₅',
    '6' => '₆',
    '7' => '₇',
    '8' => '₈',
    '9' => '₉',
)
map_subscripts(indices) = join(INDEX_MAP[c] for c in string(indices))

name(v::Variable) = v.name

Base.isless(u::Variable, v::Variable) = isless(u.name, v.name)
Base.:(==)(x::Variable, y::Variable) = x.name == y.name
Base.show(io::IO, v::Variable) = print(io, v.name)
Base.convert(::Type{Expr}, x::Variable) = x.name
Base.convert(::Type{Expression}, x::Symbol) = Variable(x)

## variable macros

"""
    @var(args...)

Declare variables with the given and automatically create the variable bindings.

## Examples

```julia
julia> @var a b x[1:2] y[1:2,1:3]
(a, b, Variable[x₁, x₂], Variable[y₁₋₁ y₁₋₂ y₁₋₃; y₂₋₁ y₂₋₂ y₂₋₃])

julia> a
a

julia> b
b

julia> x
2-element Array{Variable,1}:
 x₁
 x₂

julia> y
2×3 Array{Variable,2}:
 y₁₋₁  y₁₋₂  y₁₋₃
 y₂₋₁  y₂₋₂  y₂₋₃
```
"""
macro var(args...)
    vars, exprs = buildvars(args; unique = false)
    :($(foldl((x, y) -> :($x; $y), exprs, init = :())); $(Expr(:tuple, esc.(vars)...)))
end

"""
    @unique_var(args...)

Declare variables and automatically create the variable bindings to the given names.
This will change the names of the variables to ensure uniqueness.

## Examples

```julia
julia> @unique_var a b
(##a#591, ##b#592)

julia> a
##a#591

julia> b
##b#592
```
"""
macro unique_var(args...)
    vars, exprs = buildvars(args; unique = true)
    :($(foldl((x, y) -> :($x; $y), exprs, init = :())); $(Expr(:tuple, esc.(vars)...)))
end

function var_array(prefix, indices...)
    map(i -> Variable(prefix, i...), Iterators.product(indices...))
end

function buildvar(var; unique::Bool = false)
    if isa(var, Symbol)
        varname = unique ? gensym(var) : var
        var, :($(esc(var)) = $Variable($"$varname"))
    else
        isa(var, Expr) || error("Expected $var to be a variable name")
        Base.Meta.isexpr(
            var,
            :ref,
        ) || error("Expected $var to be of the form varname[idxset]")
        (2 ≤ length(var.args)) || error("Expected $var to have at least one index set")
        varname = var.args[1]
        prefix = unique ? string(gensym(varname)) : string(varname)
        varname, :($(esc(varname)) = var_array($prefix, $(esc.(var.args[2:end])...)))
    end
end

function buildvars(args; unique::Bool = false)
    vars = Symbol[]
    exprs = []
    for arg in args
        var, expr = buildvar(arg; unique = unique)
        push!(vars, var)
        push!(exprs, expr)
    end
    vars, exprs
end


# CONSTANTs
struct Constant <: Expression
    value::Number
end
Base.iszero(x::Constant) = iszero(x.value)
Base.isone(x::Constant) = isone(x.value)
Base.:(==)(x::Constant, y::Constant) = x.value == y.value
Base.show(io::IO, v::Constant) = print(io, v.value)
Base.convert(::Type{Expr}, x::Constant) = :($(x.value))
Base.convert(::Type{Expression}, x::Number) = Constant(x)


# OPERATIONs
const SUPPORTED_FUNCTIONS = [:+, :-, :*, :/, :^, :identity]

struct Operation <: Expression
    func::Symbol # function name
    args::Vector{Expression}

    function Operation(func::Symbol, args::Vector{Expression})
        func ∈ SUPPORTED_FUNCTIONS || _err_unsupported_func(func)
        1 ≤ length(args) ≤ 2 || _err_wrong_nargs(length(args))
        new(func, args)
    end
end
@noinline _err_unsupported_func(func) =
    throw(ArgumentError("Function `$func` not supported."))
@noinline _err_wrong_nargs(n) =
    throw(ArgumentError("Operation can only hold 1 or 2 arguments, but $n provided."))


Operation(func::Symbol, arg::Expression) = Operation(func, Expression[arg])
Operation(func::Symbol, a::Expression, b::Expression) = Operation(func, Expression[a, b])

Base.:(==)(x::Operation, y::Operation) = (x.func === y.func) && (x.args == y.args)
Base.hash(I::Operation, h::UInt) = foldr(hash, I.args, init = hash(I.func, h))
Base.convert(::Type{Operation}, c::Constant) = Operation(:identity, c)
Base.convert(::Type{Expr}, op::Operation) = Expr(:call, op.func, convert.(Expr, op.args)...)

Base.show(io::IO, op::Operation) = print(io, flattened_expr(op))
flattened_expr(expr::Expression) = convert(Expr, expr)
function flattened_expr(O::Operation)
    # Tree shrinking
    if O.func == :+ || O.func == :*
       # Flatten tree
        args = []
        for arg in O.args
            fexpr = flattened_expr(arg)
            if arg isa Operation && arg.func === O.func
                for i = 2:length(fexpr.args)
                    push!(args, fexpr.args[i])
                end
            else
                push!(args, fexpr)
            end
        end
        return Expr(:call, O.func, args...)
    end
    Expr(:call, O.func, flattened_expr.(O.args)...)
end

Latexify.@latexrecipe function f(exprs::AbstractVector{<:Expression})
    # Set default option values.
    return flattened_expr.(exprs)
end
Latexify.@latexrecipe function f(expr::Expression)
    return flattened_expr(expr)
end


################
## Expression ##
################

for (f, arity) in [(:+, 2), (:-, 1), (:-, 2), (:*, 2), (:/, 2), (:^, 2)]
    if arity == 1
        @eval Base.$f(a::Expression) = Operation($(QuoteNode(f)), a)
    elseif arity == 2
        @eval Base.$f(a::Expression, b::Expression) = Operation($(QuoteNode(f)), a, b)
        @eval Base.$f(a::Expression, b::Number) = Base.$f(a, Constant(b))
        @eval Base.$f(a::Number, b::Expression) = Base.$f(Constant(a), b)
    end
end
Base.:(+)(a::Expression) = a
# Make + 0 or * 1 no-ops
Base.:(+)(a::Constant, b::Constant) = iszero(a) ? b : (iszero(b) ? a : Operation(:+, a, b))
Base.:(+)(a::Expression, b::Constant) = iszero(b) ? a : Operation(:+, a, b)
Base.:(+)(a::Constant, b::Expression) = iszero(a) ? b : Operation(:+, a, b)
Base.:(*)(a::Constant, b::Constant) = isone(a) ? b : (isone(b) ? a : Operation(:*, a, b))
Base.:(*)(a::Expression, b::Constant) = isone(b) ? a : Operation(:*, a, b)
Base.:(*)(a::Constant, b::Expression) = isone(a) ? b : Operation(:*, a, b)

Base.iszero(::Expression) = false
Base.zero(::Expression) = Constant(0)
Base.isone(::Expression) = false
Base.one(::Expression) = Constant(1)
Base.adjoint(expr::Expression) = expr
Base.transpose(expr::Expression) = expr
Base.broadcastable(v::Expression) = Ref(v)
Base.iterate(expr::Expression) = expr, 1
Base.iterate(expr::Expression, state) = nothing
dot(x::Expression, y::Expression) = x * y

function Base.convert(::Type{Expression}, ex::Expr)
    ex.head === :call || throw(ArgumentError("internal representation does not support non-call Expr"))
    return Operation(ex.args[1], convert.(Expression, ex.args[2:end])...)
end
Expression(x) = convert(Expression, x)
Base.promote_rule(::Type{<:Expression}, ::Type{<:Number}) = Expression
Base.promote_rule(::Type{<:Expression}, ::Type{Symbol}) = Expression
Base.promote_rule(::Type{<:Expression}, ::Type{Operation}) = Expression
"""
    variables(expr::Expression)
    variables(exprs::AbstractVector{<:Expression})

Obtain all variables used in the given expression.
"""
variables(op::Expression) = sort!(collect(variables!(Set{Variable}(), op)))
function variables(exprs::AbstractVector{<:Expression})
    S = Set{Variable}()
    for expr in exprs
        variables!(S, expr)
    end
    sort!(collect(S))
end

function variables!(vars::Set{Variable}, op::Operation)
    variables!.(Ref(vars), op.args)
    vars
end
variables!(vars::Set{Variable}, var::Variable) = (push!(vars, var); vars)
variables!(vars::Set{Variable}, ::Constant) = vars

"""
    subs(expr::Expression, subs::Pair{Variable,<:Expression}...)
    subs(expr::Expression, subs::Pair{AbstractArray{<:Variable},AbstractArray{<:Expression}}...)

Substitute into the given expression.

## Example

```
@var x y z

julia> subs(x^2, x => y)
y ^ 2

julia> subs(x * y, [x,y] => [x+2,y+2])
(x + 2) * (y + 2)
```
"""
subs(x::Constant, sub::Pair{Variable,<:Expression}) = x
subs(x::Variable, sub::Pair{Variable,<:Expression}) = first(sub) == x ? last(sub) : x
function subs(op::Operation, sub::Pair{Variable,<:Number})
    subs(op, first(sub) => Constant(last(sub)))
end
function subs(op::Operation, sub::Pair{Variable,<:Expression})
    Operation(op.func, Expression[subs(arg, sub) for arg in op.args])
end
function subs(exprs::AbstractArray{<:Expression}, sub::Pair{Variable,<:Expression})
    map(e -> subs(e, sub), exprs)
end
function subs(
    expr::Union{Expression,AbstractArray{<:Expression}},
    sub_pairs::Union{
        Pair{Variable,<:Union{Number,Expression}},
        Pair{<:AbstractArray{Variable},<:AbstractArray{<:Union{Number,Expression}}},
    }...,
)
    new_expr = expr
    for sub in sub_pairs
        new_expr = subs(new_expr, sub)
    end
    new_expr
end
function subs(
    expr::Union{Expression,AbstractArray{<:Expression}},
    sub_pairs::Pair{<:AbstractArray{Variable},<:AbstractArray{<:Expression}},
)
    length(first(sub_pairs)) == length(last(sub_pairs)) || error(ArgumentError("Substitution arguments don't have the same length."))

    list_of_pairs = map((k, v) -> k => v, first(sub_pairs), last(sub_pairs))
    subs(expr, list_of_pairs...)
end

"""
    evaluate(expr::Expression, subs::Pair{Variable,<:Any}...)
    evaluate(expr::Expression, subs::Pair{AbstractArray{<:Variable},AbstractArray{<:Any}}...)

Evaluate the given expression.

## Example

```
@var x y

julia> evaluate(x^2, x => 2)
4

julia> evaluate(x * y, [x,y] => [2, 3])
6
"""
function evaluate(
    expr::Union{Expression,AbstractArray{<:Expression}},
    args::Pair{<:AbstractArray{<:Variable,N},<:AbstractArray{<:Any,N}}...,
) where {N}
    D = Dict{Variable,Any}()
    for arg in args
        for (k, v) in zip(arg...)
            D[k] = v
        end
    end
    if expr isa AbstractArray
        map(e -> evaluate(e, D), expr)
    else
        evaluate(expr, D)
    end
end
function evaluate(
    expr::Union{Expression,AbstractArray{<:Expression}},
    args::Pair{<:AbstractArray{<:Variable,N},<:AbstractArray{T,N}}...,
) where {T,N}
    D = Dict{Variable,T}()
    for arg in args
        for (k, v) in zip(arg...)
            D[k] = v
        end
    end
    if expr isa AbstractArray
        map(e -> evaluate(e, D), expr)
    else
        evaluate(expr, D)
    end
end
evaluate(op::Constant, args::Dict{Variable,<:Any}) = op.value
evaluate(op::Variable, args::Dict{Variable,<:Any}) = args[op]
function evaluate(op::Operation, args::Dict{Variable,<:Any})
    if length(op.args) == 2
        a, b = evaluate(op.args[1], args), evaluate(op.args[2], args)

        if op.func == :+
            a + b
        elseif op.func == :-
            a - b
        elseif op.func == :/
            a / b
        elseif op.func == :^
            a^b
        elseif op.func == :*
            a * b
        else
            error("Unsupported func: " * string(op.func))
        end
    elseif length(op.args) == 1
        a = evaluate(op.args[1], args)
        if op.func == :identity
            a
        elseif op.func == :-
            -a
        else
            error("Unsupported func: " * string(op.func))
        end
    else
        error("Unsupported argument length")
    end
end

(op::Constant)(args...) = evaluate(op, args...)
(op::Variable)(args...) = evaluate(op, args...)
(op::Operation)(args...) = evaluate(op, args...)

function det(A::AbstractMatrix{<:Expression})
    isequal(size(A)...) || throw(ArgumentError("Cannot compute `det` of a non-square matrix."))
    n = size(A, 1)
    n < 4 || throw(ArgumentError("`det` only supported for at most 3 by 3 matrices of `Expression`s."))

    n == 1 && return A[1, 1]
    n == 2 && return A[1, 1] * A[2, 2] - A[1, 2] * A[2, 1]
    # n == 3
    A[1, 1] * (A[2, 2] * A[3, 3] - A[2, 3] * A[3, 2]) -
    A[1, 2] * (A[2, 1] * A[3, 3] - A[2, 3] * A[3, 1]) +
    A[1, 3] * (A[2, 1] * A[3, 2] - A[2, 2] * A[3, 1])
end

"""
    simplify(expr::Expression)

Try to simplify the given expression.

```julia
julia> @var x;
julia> simplify(x + 0)
x
"""
simplify(expr::Expression) = expr
function simplify(op::Operation)
    if op.args[1] isa Operation
        u = simplify(op.args[1])
    else
        u = op.args[1]
    end
    if length(op.args) == 1
        if op.func == :identity
            return u
        end
        return Operation(op.func, u)
    end

    if op.args[2] isa Operation
        v = simplify(op.args[2])
    else
        v = op.args[2]
    end
    if op.func == :*
        if iszero(u) || iszero(v)
            return Constant(0)
        elseif isone(u)
            return v
        elseif isone(v)
            return u
        elseif u isa Constant && v isa Constant
            return Constant(u.value * v.value)
        elseif u isa Constant && v isa Operation && v.func == :*
            if v.args[1] isa Constant
                return Operation(:*, Constant(u.value * v.args[1].value), v.args[2])
            elseif v.args[2] isa Constant
                return Operation(:*, Constant(u.value * v.args[2].value), v.args[1])
            end
        elseif u isa Operation && u.func == :* && v isa Constant
            if u.args[1] isa Constant
                return Operation(:*, Constant(v.value * u.args[1].value), u.args[2])
            elseif u.args[2] isa Constant
                return Operation(:*, Constant(v.value * u.args[2].value), u.args[1])
            end
        elseif u isa Operation &&
               u.func == :* &&
               u.args[1] isa Constant &&
               v isa Operation &&
               v.func == :* &&
               v.args[1] isa Constant &&
               length(u.args) == length(v.args) == 2 && u.args[2] == v.args[2]
            return Operation(:*, Constant(u.args[1].value * v.args[1].value), u.args[2])
        elseif v isa Constant && !(u isa Constant)
            return Operation(:*, v, u)
        end
    elseif op.func == :+
        if u isa Constant && v isa Constant
            return Constant(u.value + v.value)
        elseif iszero(u)
            return v
        elseif iszero(v)
            return u
        elseif u == v
            return Operation(:*, Constant(2), v)
        elseif u isa Operation &&
               u.func == :* &&
               u.args[1] isa Constant &&
               v isa Operation &&
               v.func == :* &&
               v.args[1] isa Constant &&
               length(u.args) == length(v.args) == 2 && u.args[2] == v.args[2]
            return Operation(:*, Constant(u.args[1].value + v.args[1].value), u.args[2])
        end
    elseif op.func == :-
        if u isa Constant && v isa Constant
            return Constant(u.value - v.value)
        elseif iszero(v)
            return u
        end
    elseif op.func == :^
        if iszero(v)
            return Constant(1)
        elseif isone(v)
            return u
        end
    end
    Operation(op.func, u, v)
end

"""
    differentiate(expr::Expression, var::Variable)
    differentiate(expr::Expression, var::Vector{Variable})
    differentiate(expr::Vector{<:Expression}, var::Vector{Variable})

Compute the derivative of `expr` with respect to the given variable `var`.
"""
function differentiate(op::Operation, var::Variable)
    func, args = op.func, op.args

    # arity 1
    if length(args) == 1
        u = args[1]
        u′ = differentiate(u, var)
        if func == :-
            d = -u′
        elseif func == :identity
            d = u′
        else
            error("Unsupported arity 1 function; $func")
        end
    # arity 2
    elseif length(args) == 2
        u, v = args
        u′, v′ = differentiate(u, var), differentiate(v, var)
        # (a * b)' =  a'b + ab'
        if func === :*
            d = u * v′ + v * u′
        elseif func === :+
            d = u′ + v′
        elseif func === :-
            d = u′ - v′
        elseif func === :^
            # @assert v isa Constant
            d = v * u^(v - Constant(1)) * u′
        elseif func === :/
            d = u′ / v - (u * v′) / (v^2)
        end
    end
    simplify(d)
end

differentiate(arg::Variable, var::Variable) = arg == var ? Constant(1) : Constant(0)
differentiate(arg::Constant, var::Variable) = Constant(0)

differentiate(expr::Expression, vars::AbstractVector{Variable}) = differentiate.(expr, vars)
function differentiate(exprs::AbstractVector{<:Expression}, vars::AbstractVector{Variable})
    [differentiate(expr, v) for expr in exprs, v in vars]
end

"""
    monomials(vars, d; homogeneous::Bool = false)

Create all monomials of a given degree.

```
julia> @var x y
(x, y)

julia> monomials([x,y], 2)
6-element Array{Expression,1}:
 x ^ 2
 x * y
 y ^ 2
 x
 y
 1

julia> monomials([x,y], 2; homogeneous = true)
3-element Array{Operation,1}:
 x ^ 2
 x * y
 y ^ 2
 ```
"""
function monomials(vars::AbstractVector{Variable}, d::Integer; homogeneous::Bool = false)
    n = length(vars)
    if homogeneous
        pred = x -> sum(x) == d
    else
        pred = x -> sum(x) ≤ d
    end
    exps = collect(Iterators.filter(pred, Iterators.product(Iterators.repeated(0:d, n)...)))
    sort!(exps, lt = td_order, rev = true)
    map(exps) do exp
        simplify(prod(i -> vars[i]^exp[i], 1:n))
    end
end
function td_order(x, y)
    sx = sum(x)
    sy = sum(y)
    sx == sy ? x < y : sx < sy
end

############
## System ##
############

struct System
    expressions::Vector{Expression}
    variables::Vector{Variable}
    parameters::Vector{Variable}

    function System(
        exprs::Vector{Expression},
        vars::Vector{Variable},
        params::Vector{Variable},
    )
        check_vars_params(exprs, vars, params)
        new(exprs, vars, params)
    end
end

function check_vars_params(f, vars, params)
    vars_params = params === nothing ? vars : [vars; params]
    Δ = setdiff(variables(f), vars_params)
    isempty(Δ) || throw(ArgumentError("Not all variables or parameters of the system are given. Missing: " *
                                      join(Δ, ", ")))
    nothing
end

function System(
    exprs::Vector{<:Expression},
    variables::Vector{Variable},
    parameters::Vector{Variable} = Variable[],
)
    System(convert(Vector{Expression}, exprs), variables, parameters)
end

evaluate(F::System, x::AbstractVector) = evaluate(F.expressions, F.variables => x)
function evaluate(F::System, x::AbstractVector, p::AbstractVector)
    evaluate(F.expressions, F.variables => x, F.parameters => p)
end
(F::System)(x::AbstractVector) = evaluate(F, x)
(F::System)(x::AbstractVector, p::AbstractVector) = evaluate(F, x, p)

function Base.:(==)(F::System, G::System)
    F.expressions == G.expressions &&
    F.variables == G.variables && F.parameters == G.parameters
end

Base.size(F::System) = (length(F.expressions), length(F.variables))
Base.size(F::System, i::Integer) = size(F)[i]
Base.length(F::System) = length(F.expressions)

##############
## Homotopy ##
##############
struct Homotopy
    expressions::Vector{Expression}
    variables::Vector{Variable}
    homotopy_var::Variable
    parameters::Vector{Variable}

    function Homotopy(
        exprs::Vector{Expression},
        vars::Vector{Variable},
        homotopy_var::Variable,
        params::Vector{Variable},
    )
        check_vars_params(exprs, [vars; homotopy_var], params)
        new(exprs, vars, homotopy_var, params)
    end
end


function Homotopy(
    exprs::Vector{<:Expression},
    variables::Vector{Variable},
    homotopy_var::Variable,
    parameters::Vector{Variable} = Variable[],
)
    Homotopy(convert(Vector{Expression}, exprs), variables, homotopy_var, parameters)
end

evaluate(F::Homotopy, x::AbstractVector, t) =
    evaluate(F.expressions, F.variables => x, F.homotopy_var => t)
function evaluate(F::Homotopy, x::AbstractVector, t, p::AbstractVector)
    evaluate(F.expressions, F.variables => x, F.homotopy_var => t, F.parameters => p)
end
(F::Homotopy)(x::AbstractVector, t) = evaluate(F, x, t)
(F::Homotopy)(x::AbstractVector, t, p::AbstractVector) = evaluate(F, x, t, p)

function Base.:(==)(F::Homotopy, G::Homotopy)
    F.expressions == G.expressions &&
    F.variables == G.variables && F.parameters == G.parameters
end

Base.size(F::Homotopy) = (length(F.expressions), length(F.variables))
Base.size(F::Homotopy, i::Integer) = size(F)[i]
Base.length(F::Homotopy) = length(F.expressions)
