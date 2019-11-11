########################
## Lift to Type Level ##
########################

abstract type TExpression end
struct TOperation{F,A1,A2} <: TExpression end

TExpression(x) = convert(TExpression, x)
Base.convert(::Type{TExpression}, v::Variable) = v.name
Base.convert(::Type{TExpression}, x::Constant) = x.value
function Base.convert(::Type{TExpression}, op::Operation)
    if length(op.args) == 1
        TOperation{op.func,TExpression(op.args[1]),Nothing}()
    else
        TOperation{op.func,TExpression(op.args[1]),TExpression(op.args[2])}()
    end
end

Base.convert(::Type{Expression}, ::TOperation{F,A1,Nothing}) where {F,A1} =
    Operation(F, Expression[Expression(A1)])
Base.convert(::Type{Expression}, ::TOperation{F,A1,A2}) where {F,A1,A2} =
    Operation(F, Expression[Expression(A1), Expression(A2)])



struct TSystem{TE,V,P} end

TSystem(sys::System) = TSystem(sys.expressions, sys.variables, sys.parameters)
function TSystem(
    exprs::Vector{<:Expression},
    var_order::AbstractVector{<:Variable},
    param_order::AbstractVector{<:Variable} = Variable[],
)
    V = tuple((var.name for var in var_order)...)
    if !isempty(param_order)
        P = tuple((var.name for var in param_order)...)
    else
        P = Nothing
    end
    TE = tuple(convert.(TExpression, exprs)...)
    TSystem{TE,V,P}()
end

Base.show(io::IO, ::Type{T}) where {T<:TSystem} = show_info(io, T)
function Base.show(io::IO, TS::TSystem)
    show_info(io, typeof(TS))
    print(io, "()")
end
function show_info(io::IO, ::Type{TSystem{TE,V,P}}) where {TE,V,P}
    n = length(TE)
    m = length(V)
    mp = (P == Nothing ? 0 : length(P))
    print(io, "TSystem{$n,$m,$mp,#$(hash(TE))}")
end


System(TS::TSystem) = System(typeof(TS))
function System(::Type{TSystem{TE,V,P}}) where {TE,V,P}
    exprs = [Expression(e) for e in TE]
    vars = [Variable(e) for e in V]
    if P == Nothing
        params = Variable[]
    else
        params = [Variable(v) for v in P]
    end
    System(exprs, vars, params)
end

###########################
## CODEGEN + COMPILATION ##
###########################


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


function evaluate end
function evaluate! end
function evaluate_gradient end
function evalute_jacobian end
function evaluate_jacobian! end
function jacobian end
function jacobian! end
function dt! end
function dt end
function jacobian_dt! end


struct CompiledSystem{T<:TSystem}
    system::System
end
CompiledSystem(sys::System) = CompiledSystem{typeof(TSystem(sys))}(sys)

Base.length(F::CompiledSystem) = length(F.system)
Base.size(F::CompiledSystem) = size(F.system)
Base.size(F::CompiledSystem, i::Integer) = size(F.system, i)


function jacobian!(U::AbstractMatrix, S::CompiledSystem, args...)
    evaluate_jacobian!(nothing, U, S, args...)
    U
end
function evaluate_jacobian(S::CompiledSystem, args...)
    evaluate(S, args...), jacobian(S, args...)
end


function make_indexing(vars, params)
    lhs = Any[convert(Expr, v) for v in vars]
    rhs = Any[:(x[$i]) for i = 1:length(vars)]
    if params !== nothing
        append!(lhs, convert.(Expr, params))
        for i = 1:length(params)
            push!(rhs, :(p[$i]))
        end
    end
    :($(Expr(:tuple, lhs...)) = $(Expr(:tuple, rhs...)))
end


function _evaluate_impl(::Type{T}) where {T<:TSystem}
    sys = System(T)
    indexing = make_indexing(sys.variables, sys.parameters)
    quote
        let $indexing
            $(begin
                list = InstructionList()
                ids = [convert(Expr, push!(list, fi)) for fi in sys.expressions]
                quote
                    $(convert(Expr, list))
                    @SVector $(Expr(:vect, ids...))
                end
            end)
        end
    end
end

@generated function evaluate(F::CompiledSystem{T}, x, p = nothing) where {T}
    _evaluate_impl(T)
end

function _evaluate!_impl(::Type{T}) where {T<:TSystem}
    u = gensym(:u)
    sys = System(T)
    indexing = make_indexing(sys.variables, sys.parameters)
    quote
        let $u = u
            let $indexing
                $(begin
                    list = InstructionList()
                    ids = [convert(Expr, push!(list, fi)) for fi in sys.expressions]
                    quote
                        $(convert(Expr, list))
                        $(map(i -> :($u[$i] = $(ids[i])), 1:length(ids))...)
                    end
                end)
            end
        end
        u
    end
end

@generated function evaluate!(u, F::CompiledSystem{T}, x, p = nothing) where {T}
    _evaluate!_impl(T)
end

function _evaluate_jacobian!_impl(::Type{T}) where {T<:TSystem}
    u, U = gensym(:u), gensym(:U)
    sys = System(T)
    indexing = make_indexing(sys.variables, sys.parameters)
    n, m = size(sys)
    quote
        let ($u, $U) = (u, U)
            let $indexing
                $(begin
                    list = InstructionList()
                    eval_ids = Vector{Any}(undef, n)
                    jac_ids = Matrix{Any}(undef, n, m)
                    for i = 1:n
                        fᵢ = sys.expressions[i]
                        eval_ids[i] = convert(Expr, push!(list, fᵢ))
                        for j = 1:m
                            jac_ids[i, j] = convert(
                                Expr,
                                push!(list, differentiate(fᵢ, sys.variables[j])),
                            )
                        end
                    end
                    quote
                        $(convert(Expr, list))
                        if !($u isa Nothing)
                            $(map(i -> :($u[$i] = $(eval_ids[i])), 1:n)...)
                        end
                        $(vec([:($U[$i, $j] = $(jac_ids[i, j])) for i = 1:n, j = 1:m])...)
                        nothing
                    end
                end)
            end
        end
    end
end

@generated function evaluate_jacobian!(u, U, F::CompiledSystem{T}, x, p = nothing) where {T}
    _evaluate_jacobian!_impl(T)
end


function _jacobian_impl(::Type{T}) where {T<:TSystem}
    sys = System(T)
    indexing = make_indexing(sys.variables, sys.parameters)
    n, m = size(sys)
    quote
        let $indexing
            $(begin
                list = InstructionList()
                jac_ids = [convert(Expr, push!(list, differentiate(fᵢ, v))) for fᵢ in sys.expressions, v in sys.variables]
                quote
                    $(convert(Expr, list))
                    @SMatrix $(Expr(:vcat, map(1:n) do i
                        Expr(:row, jac_ids[i, :]...)
                    end...))
                end
            end)
        end
    end
end

@generated function jacobian(F::CompiledSystem{T}, x, p = nothing) where {T}
    _jacobian_impl(T)
end



# function compile(
#     f::Vector{Operation},
#     vars::Vector{Variable},
#     params::Union{Nothing,Vector{Variable}} = nothing,
# )
#     f = simplify.(f)
#     check_vars_params(f, vars, params)
#
#     name = Symbol("CompiledSystem##", foldr(hash, f; init = UInt(0)))
#     indexing = make_indexing(vars, params)
#     func_args = [:(x::AbstractVector)]
#     params !== nothing && push!(func_args, :(p::AbstractVector))
#
#     u = gensym(:u)
#     U = gensym(:U)
#
#     compiled = @eval begin
#         struct $name <: CompiledSystem
#             op::Vector{Operation}
#             vars::Vector{Variable}
#             params::Union{Nothing,Vector{Variable}}
#         end
#
#         function _evaluate!($u::AbstractVector, ::$name, $(func_args...))
#             let $indexing
#                 $(begin
#                     list = InstructionList()
#                     ids = [convert(Expr, push!(list, fi)) for fi in f]
#                     quote
#                         $(convert(Expr, list))
#                         $(map(i -> :($u[$i] = $(ids[i])), 1:length(ids))...)
#                         0
#                     end
#                 end)
#             end
#         end
#
#         function evaluate(::$name, $(func_args...))
#             let $indexing
#                 $(begin
#                     list = InstructionList()
#                     ids = [convert(Expr, push!(list, fi)) for fi in f]
#                     quote
#                         $(convert(Expr, list))
#                         $(Expr(:vect, ids...))
#                     end
#                 end)
#             end
#         end
#
#         function evaluate_jacobian!(
#             $u::Union{AbstractVector,Nothing},
#             $U::AbstractMatrix,
#             ::$name,
#             $(func_args...),
#         )
#             let $indexing
#                 $(begin
#                     list = InstructionList()
#                     eval_ids = [convert(Expr, push!(list, fi)) for fi in f]
#                     jac_ids = [convert(Expr, push!(list, differentiate(fi, v))) for fi in f, v in vars]
#                     quote
#                         $(convert(Expr, list))
#                         if !($u isa Nothing)
#                             $(map(i -> :($u[$i] = $(eval_ids[i])), 1:length(eval_ids))...)
#                         end
#                         $(vec([:($U[$i, $j] = $(jac_ids[i, j])) for i = 1:length(f), j = 1:length(vars)])...)
#                         nothing
#                     end
#                 end)
#             end
#         end
#
#         function jacobian(::$name, $(func_args...))
#             let $indexing
#                 $(begin
#                     list = InstructionList()
#                     jac_ids = [convert(Expr, push!(list, differentiate(fi, v))) for fi in f, v in vars]
#                     quote
#                         $(convert(Expr, list))
#                         $(Expr(:vcat, map(1:length(f)) do i
#                             Expr(:row, jac_ids[i, :]...)
#                         end...))
#                     end
#                 end)
#             end
#         end
#
#         $name
#     end
#     Base.invokelatest(compiled, f, vars, params)
# end



#
# @inline @generated function evaluate!(
#     u::U,
#     F::S,
#     x::X,
# ) where {U<:AbstractVector,S<:CompiledSystem,X<:AbstractVector}
#     quote
#         # c_func = @cfunction($_evaluate!, Int, ($U, $CPS, $X))
#         cfunc = $(Expr(:cfunction, Base.CFunction, :(HCModelKit._evaluate!), :Int, :(Core.svec(U, S, X)), :(:ccall)))
#         ccall(cfunc.ptr, Int, ($U, $S, $X), u, F, x)
#         u
#     end
# end

# Homotopies
# abstract type CompiledOperationHomotopy end
#
# Base.length(F::CompiledOperationHomotopy) = size(F, 1)
# Base.size(F::CompiledOperationHomotopy) = (length(F.op), length(F.vars))
# Base.size(F::CompiledOperationHomotopy, i::Integer) = size(F)[i]
# function jacobian!(U::AbstractMatrix, S::CompiledOperationHomotopy, args...)
#     evaluate_jacobian!(nothing, U, S, args...)
#     U
# end
# function evaluate_jacobian(S::CompiledOperationHomotopy, args...)
#     evaluate(S, args...), jacobian(S, args...)
# end
#
# function compile(
#     f::Vector{Operation},
#     vars::Vector{Variable},
#     homotopy_var::Variable,
#     params::Union{Nothing,Vector{Variable}} = nothing,
# )
#     f = simplify.(f)
#     check_vars_params(f, [vars; homotopy_var], params)
#
#     name = Symbol("CompiledOperationHomotopy##", foldr(hash, f; init = UInt(0)))
#     indexing = make_indexing(vars, params)
#     @show indexing
#     func_args = [:(x::AbstractVector), :($(convert(Expr, homotopy_var))::Number)]
#     params !== nothing && push!(func_args, :(p::AbstractVector))
#
#     u = gensym(:u)
#     U = gensym(:U)
#
#     compiled = @eval begin
#         struct $name <: CompiledOperationHomotopy
#             op::Vector{Operation}
#             vars::Vector{Variable}
#             homotopy_var::Variable
#             params::Union{Nothing,Vector{Variable}}
#         end
#
#         function evaluate!($u::AbstractVector, ::$name, $(func_args...))
#             let $indexing
#                 $(begin
#                     list = InstructionList()
#                     ids = [convert(Expr, push!(list, fi)) for fi in f]
#                     quote
#                         $(convert(Expr, list))
#                         $(map(i -> :($u[$i] = $(ids[i])), 1:length(ids))...)
#                         $u
#                     end
#                 end)
#             end
#         end
#
#         function evaluate(::$name, $(func_args...))
#             let $indexing
#                 $(begin
#                     list = InstructionList()
#                     ids = [convert(Expr, push!(list, fi)) for fi in f]
#                     quote
#                         $(convert(Expr, list))
#                         $(Expr(:vect, ids...))
#                     end
#                 end)
#             end
#         end
#
#         function evaluate_jacobian!(
#             $u::Union{AbstractVector,Nothing},
#             $U::AbstractMatrix,
#             ::$name,
#             $(func_args...),
#         )
#             let $indexing
#                 $(begin
#                     list = InstructionList()
#                     eval_ids = [convert(Expr, push!(list, fi)) for fi in f]
#                     jac_ids = [convert(Expr, push!(list, differentiate(fi, v))) for fi in f, v in vars]
#                     quote
#                         $(convert(Expr, list))
#                         if !($u isa Nothing)
#                             $(map(i -> :($u[$i] = $(eval_ids[i])), 1:length(eval_ids))...)
#                         end
#                         $(vec([:($U[$i, $j] = $(jac_ids[i, j])) for i = 1:length(f), j = 1:length(vars)])...)
#                         nothing
#                     end
#                 end)
#             end
#         end
#
#         function jacobian(::$name, $(func_args...))
#             let $indexing
#                 $(begin
#                     list = InstructionList()
#                     jac_ids = [convert(Expr, push!(list, differentiate(fi, v))) for fi in f, v in vars]
#                     quote
#                         $(convert(Expr, list))
#                         $(Expr(:vcat, map(1:length(f)) do i
#                             Expr(:row, jac_ids[i, :]...)
#                         end...))
#                     end
#                 end)
#             end
#         end
#
#         function dt!($u::AbstractVector, ::$name, $(func_args...))
#             let $indexing
#                 $(begin
#                     list = InstructionList()
#                     ids = [convert(Expr, push!(list, differentiate(fi, homotopy_var))) for fi in f]
#                     quote
#                         $(convert(Expr, list))
#                         $(map(i -> :($u[$i] = $(ids[i])), 1:length(ids))...)
#                         $u
#                     end
#                 end)
#             end
#         end
#
#         function dt(::$name, $(func_args...))
#             let $indexing
#                 $(begin
#                     list = InstructionList()
#                     ids = [push!(list, differentiate(fi, homotopy_var)) for fi in f]
#                     quote
#                         $(convert(Expr, list))
#                         $(Expr(:vect, convert.(Expr, ids)...))
#                     end
#                 end)
#             end
#         end
#
#         function jacobian_dt!(
#             $U::AbstractMatrix,
#             $u::AbstractVector,
#             ::$name,
#             $(func_args...),
#         )
#             let $indexing
#                 $(begin
#                     list = InstructionList()
#                     ids = [convert(Expr, push!(list, differentiate(fi, homotopy_var))) for fi in f]
#                     jac_ids = [convert(Expr, push!(list, differentiate(fi, v))) for fi in f, v in vars]
#                     quote
#                         $(convert(Expr, list))
#                         $(map(i -> :($u[$i] = $(ids[i])), 1:length(ids))...)
#                         $(vec([:($U[$i, $j] = $(jac_ids[i, j])) for i = 1:length(f), j = 1:length(vars)])...)
#                         nothing
#                     end
#                 end)
#             end
#         end
#
#         $name
#     end
#
#     Base.invokelatest(compiled, f, vars, homotopy_var, params)
# end


# function compile(
#     f::Operation,
#     vars::AbstractVector{Variable},
#     params::Union{Nothing,AbstractVector{Variable}} = nothing,
# )
#     f = simplify(f)
#     check_vars_params(f, vars, params)
#
#     name = Symbol("CompiledOperation##", hash(f))
#     indexing = make_indexing(vars, params)
#     func_args = [:(x::AbstractVector)]
#     params !== nothing && push!(func_args, :(p::AbstractVector))
#
#     compiled = @eval begin
#         struct $name <: CompiledOperation
#             op::Operation
#             vars::Vector{Variable}
#             params::Union{Nothing,Vector{Variable}}
#         end
#
#         function evaluate(::$name, $(func_args...))
#             let $indexing
#                 $(convert(Expr, InstructionList(f)))
#             end
#         end
#
#         function evaluate_gradient(::$name, $(func_args...))
#             let $indexing
#                 $(begin
#                     list = InstructionList()
#                     f_x = convert(Expr, push!(list, f))
#                     grad_x = let
#                         ids = [push!(list, differentiate(f, v)) for v in vars]
#                         :(@SVector $(Expr(:vect, convert.(Expr, ids)...)))
#                     end
#                     quote
#                         $(convert(Expr, list))
#                         $f_x, $grad_x
#                     end
#                 end)
#             end
#         end
#         $name
#     end
#     compiled(f, vars, params)
# end
