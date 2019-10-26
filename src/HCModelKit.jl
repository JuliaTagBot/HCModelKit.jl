module HCModelKit

using OrderedCollections: OrderedDict

export Expression, Constant, Variable, Operation, Instruction, InstructionList

abstract type Expression end

const IndexMap = Dict{Char,Char}(
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
map_subscripts(indices) = join(IndexMap[c] for c in string(indices))

struct Variable <: Expression
    name::Symbol
end
Variable(s::AbstractString) = Variable(Symbol(s))
Variable(name, indices...) = Variable("$(name)$(join(map_subscripts.(indices), "₋"))")
name(v::Variable) = v.name

Base.show(io::IO, v::Variable) = print(io, v.name)


struct Constant <: Expression
    value::Number
end
Base.show(io::IO, v::Constant) = print(io, v.value)


struct Operation <: Expression
    func::Tuple{Union{Expr,Symbol},Symbol} # module, function name
    args::Vector{Expression}
end


Base.:(==)(x::Constant, y::Constant) = x.value == y.value
Base.:(==)(x::Variable, y::Variable) = x.name == y.name
Base.:(==)(x::Operation, y::Operation) = (x.func === y.func) && (x.args == y.args)

Base.convert(::Type{Expression}, x::Number) = Constant(x)
Base.convert(::Type{Expression}, x::Symbol) = Variable(x)
function Base.convert(::Type{Expression}, ex::Expr)
    ex.head === :call || throw(ArgumentError("internal representation does not support non-call Expr"))

    op = module_funcname(ex.args[1])
    args = convert.(Expression, ex.args[2:end])
    return Operation(op, args)
end
Base.convert(::Type{Expr}, x::Constant) = :($(x.value))
Base.convert(::Type{Expr}, x::Variable) = x.name
function Base.convert(::Type{Expr}, op::Operation)
    if op.func[1] == :Base
        Expr(:call, op.func[2], convert.(Expr, op.args)...)
    else
        Expr(:call, :($(op.func[1]).$(op.func[2])), convert.(Expr, op.args)...)
    end
end

function module_funcname(expr::Symbol)
    if isdefined(Base, expr)
        (:Base, expr)
    elseif isdefined(SpecialFunctions, expr)
        (:SpecialFunctions, expr)
    elseif isdefined(Statistics, expr)
        (:Statistics, expr)
    elseif isdefined(NaNMath, expr)
        (:NaNMath, expr)
    elseif isdefined(Main, expr)
        (:Main, expr)
    else
        error("Cannot find module where function $expr is defined")
    end
end
function module_funcname(expr::Expr)
    expr.head == :. || error("Unexpected format")

    if expr.args[2] isa Symbol
        funcname = expr.args[2]
    elseif expr.args[2] isa QuoteNode
        funcname = expr.args[2].value
    end
    (expr.args[1], funcname)
end


# Binary & unary operators and functions
import DiffRules, Statistics, SpecialFunctions, NaNMath

for (M, f, arity) in DiffRules.diffrules()
    fun = :($M.$f)
    if arity == 1
        @eval $M.$f(expr::Expression) = Operation($(M, f), Expression[expr])
    elseif arity == 2
        @eval $M.$f(a::Expression, b::Expression) = Operation($(M, f), Expression[a, b])
        @eval $M.$f(a::Expression, b::Number) = $M.$f(a, Constant(b))
        @eval $M.$f(a::Number, b::Expression) = $M.$f(Constant(a), b)
    end
end

function derivative(M::Symbol, f::Symbol, args::NTuple{N,Symbol}, i::Int = 1) where {N}
    # TODO: Extend for custom functions
    partials = DiffRules.diffrule(M, f, args...)
    if N == 1
        partials
    else
        partials[i]
    end
end

struct InstructionId <: Expression
    var::Symbol
    n::Int
end
Base.convert(::Type{Expr}, id::InstructionId) = Symbol(id.var, id.n)

struct Instruction
    func::Tuple{Union{Expr,Symbol},Symbol} # module, function name
    args::Vector{Union{InstructionId, Constant, Variable}}
end

Instruction(x::Variable) = Instruction((:Base, :identity), [x])
Instruction(x::Constant) = Instruction((:Base, :identity), [x])

Base.hash(I::Instruction, h::UInt) = foldr(hash, I.args, init = hash(I.func, h))

function Base.convert(::Type{Expr}, op::Instruction)
    if op.func[1] == :Base
        if op.func[2] == :identity
            op.args[1]
        else
            Expr(:call, op.func[2], convert.(Expr, op.args)...)
        end
    else
        Expr(:call, :($(op.func[1]).$(op.func[2])), convert.(Expr, op.args)...)
    end
end

Base.show(io::IO, op::Instruction) = print(io, convert(Expr, op))
Base.:(==)(a::Instruction, b::Instruction) = a.func == b.func && a.args == b.args



struct InstructionList
    instructions::OrderedDict{Instruction,InstructionId}
    derivatives::Dict{InstructionId,InstructionId}
    var::Symbol
    n::Base.RefValue{Int}
end
function InstructionList(; var::Symbol = :ι, n::Base.RefValue{Int} = Ref(0))
    instructions = OrderedDict{Instruction,InstructionId}()
    derivatives = Dict{InstructionId, InstructionId}()
    InstructionList(instructions, derivatives, var, n)
end
function InstructionList(op::Expression; kwargs...)
    list = InstructionList(; kwargs...)
    push!(list, op)
    list
end

function Base.push!(v::InstructionList, i::Instruction)
    if haskey(v.instructions, i)
        return v.instructions[i]
    else
        id = InstructionId(v.var, v.n[] += 1)
        push!(v.instructions, i => id)
        return id
    end
end
Base.push!(list::InstructionList, x::Constant) = push!(list, Instruction(x))
Base.push!(list::InstructionList, x::Variable) = push!(list, Instruction(x))
function Base.push!(list::InstructionList, op::Operation)
    args = map(op.args) do arg
        if arg isa Operation
            push!(list, arg)
        else
            arg
        end
    end

    push!(list, Instruction(op.func, args))
end

Base.length(v::InstructionList) = length(v.instructions)
Base.iterate(v::InstructionList) = iterate(v.instructions)
Base.iterate(v::InstructionList, state) = iterate(v.instructions, state)
Base.eltype(v::Type{InstructionList}) = eltype(v.instructions)
# Base.size(v::InstructionList) = (length(v),)
# Base.getindex(v::InstructionList, i) = getindex(v.instructions, i)
# Base.setindex!(v::InstructionList, instr::Instruction, i) =
#     getindex(v.instructions, instr, i)


function Base.show(io::IO, ::MIME"text/plain", list::InstructionList)
    println(io, "InstructionList:")
    for (instr, id) in list
        println(io, convert(Expr, id), " = ", convert(Expr, instr))
    end
end




function derivative!(list::InstructionList, instr::Instruction, id::InstructionId)
    if instr.func == (:Base, :*)
        derivatives = [instr.args[2], instr.args[1]]
    end
    d = chain_rule(instr.args[1], derivatives[1], list)
    for i in 2:length(derivatives)
        d += chain_rule(instr.args[i], derivatives[i], list)
    end
    d_id = push!(list, d)
    list.derivatives[id] = d_id
    list
end

chain_rule(arg::InstructionId, deriv, list::InstructionList) = deriv * list.derivatives[arg]
chain_rule(arg::Variable, deriv, list::InstructionList) = deriv
chain_rule(arg::Constant, deriv, list::InstructionList) = Constant(0)



# function derivative(op::Operation, args::NTuple{N,Variable}, i::Int = 1) where N
#     convert(Expression, diffrule_derivative(op.func..., name.(args), i))
# end


end # module
