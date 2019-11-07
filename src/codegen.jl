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


abstract type CompiledOperation end
abstract type CompiledOperationSystem end

Base.length(F::CompiledOperationSystem) = size(F, 1)
Base.size(F::CompiledOperationSystem) = (length(F.op), length(F.vars))
Base.size(F::CompiledOperationSystem, i::Integer) = size(F)[i]
function jacobian!(U::AbstractMatrix, S::CompiledOperationSystem, args...)
    evaluate_jacobian!(nothing, U, S, args...)
    U
end
function evaluate_jacobian(S::CompiledOperationSystem, args...)
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

function check_vars_params(f, vars, params)
    vars_params = params === nothing ? vars : [vars; params]
    Δ = setdiff(variables(f), vars_params)
    isempty(Δ) || throw(ArgumentError("Not all variables or parameters of the system are given. Missing: " *
                                      join(Δ, ", ")))
    nothing
end

function compile(
    f::Operation,
    vars::AbstractVector{Variable},
    params::Union{Nothing,AbstractVector{Variable}} = nothing,
)
    f = simplify(f)
    check_vars_params(f, vars, params)

    name = Symbol("CompiledOperation##", hash(f))
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
    f = simplify.(f)
    check_vars_params(f, vars, params)

    name = Symbol("CompiledOperationSystem##", foldr(hash, f; init = UInt(0)))
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


# Homotopies
abstract type CompiledOperationHomotopy end

Base.length(F::CompiledOperationHomotopy) = size(F, 1)
Base.size(F::CompiledOperationHomotopy) = (length(F.op), length(F.vars))
Base.size(F::CompiledOperationHomotopy, i::Integer) = size(F)[i]
function jacobian!(U::AbstractMatrix, S::CompiledOperationHomotopy, args...)
    evaluate_jacobian!(nothing, U, S, args...)
    U
end
function evaluate_jacobian(S::CompiledOperationHomotopy, args...)
    evaluate(S, args...), jacobian(S, args...)
end

function compile(
    f::Vector{Operation},
    vars::Vector{Variable},
    homotopy_var::Variable,
    params::Union{Nothing,Vector{Variable}} = nothing,
)
    f = simplify.(f)
    check_vars_params(f, [vars; homotopy_var], params)

    name = Symbol("CompiledOperationHomotopy##", foldr(hash, f; init = UInt(0)))
    indexing = make_indexing(vars, params)
    @show indexing
    func_args = [:(x::AbstractVector), :($(convert(Expr, homotopy_var))::Number)]
    params !== nothing && push!(func_args, :(p::AbstractVector))

    u = gensym(:u)
    U = gensym(:U)

    compiled = @eval begin
        struct $name <: CompiledOperationHomotopy
            op::Vector{Operation}
            vars::Vector{Variable}
            homotopy_var::Variable
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

        function dt!($u::AbstractVector, ::$name, $(func_args...))
            let $indexing
                $(begin
                    list = InstructionList()
                    ids = [convert(Expr, push!(list, differentiate(fi, homotopy_var))) for fi in f]
                    quote
                        $(convert(Expr, list))
                        $(map(i -> :($u[$i] = $(ids[i])), 1:length(ids))...)
                        $u
                    end
                end)
            end
        end

        function dt(::$name, $(func_args...))
            let $indexing
                $(begin
                    list = InstructionList()
                    ids = [push!(list, differentiate(fi, homotopy_var)) for fi in f]
                    quote
                        $(convert(Expr, list))
                        $(Expr(:vect, convert.(Expr, ids)...))
                    end
                end)
            end
        end

        function jacobian_dt!(
            $U::AbstractMatrix,
            $u::AbstractVector,
            ::$name,
            $(func_args...),
        )
            let $indexing
                $(begin
                    list = InstructionList()
                    ids = [convert(Expr, push!(list, differentiate(fi, homotopy_var))) for fi in f]
                    jac_ids = [convert(Expr, push!(list, differentiate(fi, v))) for fi in f, v in vars]
                    quote
                        $(convert(Expr, list))
                        $(map(i -> :($u[$i] = $(ids[i])), 1:length(ids))...)
                        $(vec([:($U[$i, $j] = $(jac_ids[i, j])) for i = 1:length(f), j = 1:length(vars)])...)
                        nothing
                    end
                end)
            end
        end

        $name
    end

    Base.invokelatest(compiled, f, vars, homotopy_var, params)
end
