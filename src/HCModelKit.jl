module HCModelKit

using OrderedCollections: OrderedDict

export Expression, Constant, Variable, Operation, Instruction, InstructionList

export differentiate, compile, evaluate, evaluate!, evaluate_gradient, @var, subs, variables

abstract type Expression end

Base.iszero(::Expression) = false
Base.isone(::Expression) = false

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

# VARIABLE
struct Variable <: Expression
    name::Symbol
end
Variable(name, indices...) = Variable("$(name)$(join(map_subscripts.(indices), "₋"))")
Variable(s::AbstractString) = Variable(Symbol(s))

name(v::Variable) = v.name

Base.isless(u::Variable, v::Variable) = isless(u.name, v.name)
Base.:(==)(x::Variable, y::Variable) = x.name == y.name
Base.show(io::IO, v::Variable) = print(io, v.name)
Base.convert(::Type{Expr}, x::Variable) = x.name
Base.convert(::Type{Expression}, x::Symbol) = Variable(x)

macro var(args...)
    vars, exprs = buildvars(args)
    :($(foldl((x, y) -> :($x; $y), exprs, init = :())); $(Expr(:tuple, esc.(vars)...)))
end

function var_array(prefix, indices...)
    map(i -> Variable(prefix, i...), Iterators.product(indices...))
end

function buildvar(var)
    if isa(var, Symbol)
        var, :($(esc(var)) = $Variable($"$var"))
    else
        isa(var, Expr) || error("Expected $var to be a variable name")
        Base.Meta.isexpr(
            var,
            :ref,
        ) || error("Expected $var to be of the form varname[idxset]")
        (2 ≤ length(var.args)) || error("Expected $var to have at least one index set")
        varname = var.args[1]
        prefix = string(varname)
        varname, :($(esc(varname)) = var_array($prefix, $(esc.(var.args[2:end])...)))
    end
end

function buildvars(args)
    vars = Symbol[]
    exprs = []
    for arg in args
        var, expr = buildvar(arg)
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
const SUPPORTED_FUNCTIONS = [:+, :-, :*, :/, :^]


struct Operation <: Expression
    func::Symbol # function name
    args::Vector{Expression}

    function Operation(func::Symbol, args::Vector{Expression})
        func ∈ SUPPORTED_FUNCTIONS || _err_unsupported_func(func)
        1 ≤ length(args) ≤ 2 || _err_wrong_nargs(n)
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
function Base.convert(::Type{Expr}, op::Operation)
    Expr(:call, op.func, convert.(Expr, op.args)...)
end
Base.show(io::IO, op::Operation) = print(io, convert(Expr, op))
Base.hash(I::Operation, h::UInt) = foldr(hash, I.args, init = hash(I.func, h))


variables(::Constant) = Variables[]
variables(v::Variable) = Variables[v]
variables(op::Operation) = sort!(collect(variables!(Set{Variable}(), op)))

function variables!(vars::Set{Variable}, op::Operation)
    variables!.(Ref(vars), op.args)
    vars
end
variables!(vars::Set{Variable}, var::Variable) = (push!(vars, var); vars)
variables!(vars::Set{Variable}, ::Constant) = vars

subs(x::Constant, sub::Pair{Variable,<:Expression}) = x
subs(x::Variable, sub::Pair{Variable,<:Expression}) = first(sub) == x ? last(sub) : x
function subs(op::Operation, sub::Pair{Variable,<:Number})
    subs(op, first(sub)=>Constant(last(sub)))
end
function subs(op::Operation, sub::Pair{Variable,<:Expression})
    Operation(op.func, Expression[subs(arg, sub) for arg in op.args])
end
function subs(
    expr::Expression,
    sub_pairs::Union{
        Pair{Variable,<:Union{Number, Expression}},
        Pair{<:AbstractArray{Variable},<:AbstractArray{<:Union{Number, Expression}}},
    }...,
)
    new_expr = expr
    for sub in sub_pairs
        new_expr = subs(new_expr, sub)
    end
    new_expr
end
function subs(
    expr::Expression,
    sub_pairs::Pair{<:AbstractArray{Variable},<:AbstractArray{<:Expression}},
)
    length(first(sub_pairs)) == length(last(sub_pairs)) || error(ArgumentError("Substitution arguments don't have the same length."))

    list_of_pairs = map((k, v) -> k => v, first(sub_pairs), last(sub_pairs))
    subs(expr, list_of_pairs...)
end


Base.broadcastable(v::Expression) = Ref(v)
function Base.convert(::Type{Expression}, ex::Expr)
    ex.head === :call || throw(ArgumentError("internal representation does not support non-call Expr"))
    return Operation(ex.args[1], convert.(Expression, ex.args[2:end])...)
end

for (f, arity) in [(:+, 2), (:-, 1), (:-, 2), (:*, 2), (:/, 2), (:^, 2)]
    if arity == 1
        @eval Base.$f(a::Expression) = Operation($(QuoteNode(f)), a)
    elseif arity == 2
        @eval Base.$f(a::Expression, b::Expression) = Operation($(QuoteNode(f)), a, b)
        @eval Base.$f(a::Expression, b::Number) = Operation($(QuoteNode(f)), a, Constant(b))
        @eval Base.$f(a::Number, b::Expression) = Operation($(QuoteNode(f)), Constant(a), b)
    end
end
Base.:(+)(a::Expression) = a

simplify(expr::Expression) = expr
function simplify(op::Operation)
    if op.args[1] isa Operation
        u = simplify(op.args[1])
    else
        u = op.args[1]
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


function differentiate(op::Operation, var::Variable)
    func, args = op.func, op.args

    # arity 1
    if length(args) == 1
        u = args[1]
        u′ = differentiate(u, var)
        if func == :-
            d = -u′
        elseif func === nothing
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


#
# Instruction Id
struct InstructionId
    name::Symbol
end
name(id::InstructionId) = id.name

Base.:(==)(x::InstructionId, y::InstructionId) = x.name == y.name
Base.show(io::IO, v::InstructionId) = print(io, v.name)
Base.convert(::Type{Expr}, x::InstructionId) = x.name

const SimpleExpression = Union{InstructionId,Constant,Variable}

struct Instruction
    func::Union{Nothing,Symbol} # module, function name
    args::Vector{SimpleExpression}
end

Instruction(f::Union{Nothing,Symbol}, a::SimpleExpression) = Instruction(f, [a])
Instruction(x::SimpleExpression) = Instruction(nothing, x)

Base.hash(I::Instruction, h::UInt) = foldr(hash, I.args, init = hash(I.func, h))
Base.:(==)(a::Instruction, b::Instruction) = a.func == b.func && a.args == b.args

function Base.convert(::Type{Expr}, op::Instruction)
    if op.func === nothing && length(op.args) == 1
        convert(Expr, op.args[1])
    else
        Expr(:call, op.func, convert.(Expr, op.args)...)
    end
end
Base.show(io::IO, op::Instruction) = print(io, convert(Expr, op))


struct InstructionList
    instructions::OrderedDict{Instruction,InstructionId}
    var::Symbol
    n::Base.RefValue{Int}
end

function InstructionList(; var::Symbol = :ι, n::Base.RefValue{Int} = Ref(0))
    instructions = OrderedDict{Instruction,InstructionId}()
    InstructionList(instructions, var, n)
end
function InstructionList(op::Expression; kwargs...)
    list = InstructionList(; kwargs...)
    push!(list, op)
    list
end

function Base.push!(v::InstructionList, i::Instruction, id::InstructionId)
    if haskey(v.instructions, i)
        push!(v.instructions, Instruction(v.instructions[i]) => id)
    else
        push!(v.instructions, i => id)
    end
end

function Base.push!(v::InstructionList, i::Instruction; id_var::Symbol = v.var)
    if haskey(v.instructions, i)
        v.instructions[i]
    else
        id = InstructionId(Symbol(id_var, v.n[] += 1))
        push!(v.instructions, i => id)
        id
    end
end

Base.push!(list::InstructionList, x::SimpleExpression) = push!(list, Instruction(x))

function Base.push!(list::InstructionList, op::Operation)
    if op.func == :^ &&
       op.args[2] isa Constant && op.args[2].value isa Integer && op.args[2].value > 1
        n = op.args[2].value
        if n == 2
            i = _push!(list, op.args[1])
            push!(list, Instruction(:*, [i, i]))
        else
            n₁ = div(n, 2) + 1 # make sure n₁ > n₂
            n₂ = n - n₁
            id1 = push!(list, Operation(:^, op.args[1], Constant(n₁)))
            if n₂ == 1
                push!(list, Instruction(:*, [id1, _push!(list, op.args[1])]))
            else
                id2 = push!(list, Operation(:^, op.args[1], Constant(n₂)))
                push!(list, Instruction(:*, [id1, id2]))
            end
        end
    else
        push!(list, Instruction(op.func, map(arg -> _push!(list, arg), op.args)))
    end
end
_push!(list::InstructionList, op::Operation) = push!(list, op)
_push!(list::InstructionList, op::Expression) = op

Base.length(v::InstructionList) = length(v.instructions)
Base.iterate(v::InstructionList) = iterate(v.instructions)
Base.iterate(v::InstructionList, state) = iterate(v.instructions, state)
Base.eltype(v::Type{InstructionList}) = eltype(v.instructions)

function Base.show(io::IO, ::MIME"text/plain", list::InstructionList)
    println(io, "InstructionList:")
    for (instr, id) in list
        println(io, convert(Expr, id), " = ", convert(Expr, instr))
    end
end

function Base.convert(::Type{Expr}, list::InstructionList)
    Expr(:block, map(list) do (instr, id)
        :($(convert(Expr, id)) = $(convert(Expr, instr)))
    end...)
end


## CODEGEN

abstract type CompiledOperation end
abstract type CompiledOperationSystem end

using StaticArrays

function evaluate end
function evaluate! end
function evaluate_gradient end
function evaluate_jacobian! end

function make_indexing(vars, params)
    lhs = convert.(Expr, vars)
    rhs = [:(x[$i]) for i = 1:length(vars)]
    if params !== nothing
        append!(lhs, convert.(Expr, vars))
        append!(rhs, convert.(Expr, params))
    end
    :($(Expr(:tuple, lhs...)) = $(Expr(:tuple, rhs...)))
end

function compile(
    f::Operation,
    vars::AbstractVector{Variable},
    params::Union{Nothing,AbstractVector{Variable}} = nothing,
)
    name = Symbol(:CompiledOperation2__, hash(f))
    indexing = make_indexing(vars, params)
    func_args = [:(x::AbstractVector)]
    params !== nothing && push!(func_args, :(p::AbstractVector))

    compiled = @eval begin
        struct $name <: CompiledOperation
            op::Operation
            vars::Vector{Variable}
            params::Union{Nothing,Vector{Variable}}
        end

        function evaluate(::$name, $(func_args...))
            let $indexing
                $(convert(Expr, InstructionList(f)))
            end
        end

        function evaluate_gradient(::$name, $(func_args...))
            let $indexing
                $(begin
                    list = InstructionList()
                    f_x = convert(Expr, push!(list, f))
                    grad_x = let
                        ids = [push!(list, differentiate(f, v)) for v in vars]
                        :(@SVector $(Expr(:vect, convert.(Expr, ids)...)))
                    end
                    quote
                        $(convert(Expr, list))
                        $f_x, $grad_x
                    end
                end)
            end
        end
        $name
    end
    compiled(f, vars, params)
end

function compile(
    f::Vector{Operation},
    vars::Vector{Variable},
    params::Union{Nothing,Vector{Variable}} = nothing,
)
    name = Symbol(:CompiledOperationSystem_, foldr(hash, f; init = UInt(0)))
    indexing = make_indexing(vars, params)
    func_args = [:(x::AbstractVector)]
    params !== nothing && push!(func_args, :(p::AbstractVector))
    u = gensym(:u)
    U = gensym(:U)

    compiled = @eval begin
        struct $name <: CompiledOperationSystem
            op::Vector{Operation}
            vars::Vector{Variable}
            params::Union{Nothing,Vector{Variable}}
        end

        function evaluate!($u::AbstractVector, ::$name, $(func_args...))
            let $indexing
                $(begin
                    list = InstructionList()
                    ids = [convert(Expr, push!(list, fi)) for fi in f]
                    quote
                        $(convert(Expr, list))
                        $(map(i -> :($u[$i] = $(ids[i])), 1:length(ids))...)
                        $u
                    end
                end)
            end
        end

        function evaluate(::$name, $(func_args...))
            let $indexing
                $(begin
                    list = InstructionList()
                    ids = [convert(Expr, push!(list, fi)) for fi in f]
                    quote
                        $(convert(Expr, list))
                        $(Expr(:vect, ids...))
                    end
                end)
            end
        end

        function evaluate_jacobian!(
            $u::Union{AbstractVector,Nothing},
            $U::AbstractMatrix,
            ::$name,
            $(func_args...),
        )
            let $indexing
                $(begin
                    list = InstructionList()
                    eval_ids = [convert(Expr, push!(list, fi)) for fi in f]
                    jac_ids = [convert(Expr, push!(list, differentiate(fi, v))) for fi in f, v in vars]
                    quote
                        $(convert(Expr, list))
                        if !($u isa Nothing)
                            $(map(i -> :($u[$i] = $(eval_ids[i])), 1:length(eval_ids))...)
                        end
                        $(vec([:($U[$i, $j] = $(jac_ids[i, j])) for i = 1:length(f), j = 1:length(vars)])...)
                        nothing
                    end
                end)
            end
        end

        function jacobian(::$name, $(func_args...))
            let $indexing
                $(begin
                    list = InstructionList()
                    jac_ids = [convert(Expr, push!(list, differentiate(fi, v))) for fi in f, v in vars]
                    quote
                        $(convert(Expr, list))
                        $(Expr(:vcat, map(1:length(f)) do i
                            Expr(:row, jac_ids[i, :]...)
                        end...))
                    end
                end)
            end
        end

        $name
    end
    Base.invokelatest(compiled, f, vars, params)
end

Base.size(F::CompiledOperationSystem) = (length(F.op), length(F.vars))
function jacobian!(U::AbstractMatrix, S::CompiledOperationSystem, args...)
    jacobian!(nothing, U, S, args...)
end


end # module
