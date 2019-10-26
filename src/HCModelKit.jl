module HCModelKit

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



struct Instruction
    func::Tuple{Union{Expr,Symbol},Symbol} # module, function name
    args::Vector{Any}
end

Instruction(x::Variable) = Instruction((:Base, :identity), [x.name])
Instruction(x::Constant) = Instruction((:Base, :identity), [x.value])

function Base.convert(::Type{Expr}, op::Instruction)
    if op.func[1] == :Base
        if op.func[2] == :identity
            op.args[1]
        else
            Expr(:call, op.func[2], op.args...)
        end
    else
        Expr(:call, :($(op.func[1]).$(op.func[2])), op.args...)
    end
end
Base.show(io::IO, op::Instruction) = print(io, convert(Expr, op))
Base.:(==)(a::Instruction, b::Instruction) = a.func == b.func && a.args == b.args

struct InstructionList <: AbstractVector{Instruction}
    instructions::Vector{Instruction}
    var::Symbol
end

Base.push!(v::InstructionList, i::Instruction) = push!(v.instructions, i)
Base.length(v::InstructionList) = length(v.instructions)
Base.size(v::InstructionList) = (length(v),)
Base.getindex(v::InstructionList, i) = getindex(v.instructions, i)
Base.setindex!(v::InstructionList, instr::Instruction, i) =
    getindex(v.instructions, instr, i)

function InstructionList(op::Expression, variables::Vector{Variable}, var::Symbol)
    list = InstructionList(Instruction.(variables), var)
    instruction_list!(list, op)
    list
end

function instruction_list!(list::InstructionList, op::Operation)
    args = map(op.args) do arg
        if arg isa Operation
            instruction_list!(list, arg)
            Symbol(list.var, length(list))
        elseif arg isa Variable
            arg.name
        elseif arg isa Constant
            arg.value
        end
    end

    push!(list, Instruction(op.func, args))
end

function Base.show(io::IO, ::MIME"text/plain", list::InstructionList)
    println(io, "InstructionList:")
    N = length(list)
    for i = 1:N
        print(io, list.var, i, " = ", convert(Expr, list[i]))
        if i < N
            println(io)
        end
    end
end


# function derivative(op::Operation, args::NTuple{N,Variable}, i::Int = 1) where N
#     convert(Expression, diffrule_derivative(op.func..., name.(args), i))
# end


end # module
