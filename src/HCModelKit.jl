module HCModelKit

export Expression, Constant, Variable, Operation

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

Base.show(io::IO, v::Variable) = print(io, v.name)


dump(:(Base.Threads.identity([12, 32])))

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
    expr.head == :. || error("Unexpected format")

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
    for i = 1:arity
        @eval function derivative(::typeof($M.$f), args::NTuple{$arity,Any}, ::Val{$i})
            M2, f2 = $(M, f)
            partials = DiffRules.diffrule(M2, f2, args...)
            dx = @static $arity == 1 ? partials : partials[$i]
            convert(Expression, dx)
        end
    end
end


end # module
