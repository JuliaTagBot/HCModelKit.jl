using Pkg;
Pkg.activate(@__DIR__)

using HCModelKit, BenchmarkTools


@var a b c[1:20]

(a + b)^32

@var x y z

f = (2x+3y-3)^2 + 7z+3
g = (x - y^2+3)^2 + 3x*y+z^2




@var a b

subs(f, x => a^4, y => a)
subs(f,  z => 3)
subs(f, [x, y] => [a, b])
subs(f, [x, y] => [a, b], z => 2)

sub_pairs = [x, y] => [a, b]

map((k,v) -> k => v, first(sub_pairs), last(sub_pairs))


sys = compile([f, g], [x, y, z])
evaluate(sys, [2, 32, 1])
HCModelKit.jacobian(sys, [2, 32, 1])


evaluate(sys, [3.123, 0.412212, 3.12])
u = zeros(ComplexF64, 2)
U = zeros(ComplexF64, 2, 3)
v = randn(ComplexF64, 3)
@btime HCModelKit.evaluate_and_jacobian!($u, $U, $sys, $v)
@btime HCModelKit.evaluate!($u, $sys, $v)

HCModelKit.evaluate_jacobian!(nothing, U, sys, v)
HCModelKit.evaluate_and_jacobian!($u, $U, $sys, $v)

u
U

v = rand(2)
@btime evaluate!($u, $sys, $v)









evaluate_gradient(sys, [3.123, 0.412212])

g2 = ((x - y^2+3)^2 + 3x*y)^2
sys2 = HCModelKit.compile_system([f, g2], [x, y])

evaluate(sys2, [3.123, 0.412212])
u = rand(2)
v = rand(2)
@btime HCModelKit.evaluate!($u, $sys, $v)


differentiate(differentiate(differentiate(differentiate((2x+3y)^2, x), x), x), x)

f = ((2x+3y-3)^2 + 3)^3

differentiate.(f, [x, y])
convert(Expr, InstructionList(f))



name = Symbol(:gen_struct__, hash(f))
vars = [x,y]

indexing = Expr(:tuple, (:(x[$i]) for i in 1:length(vars))...)




sys = compile(f, [x, y])

evaluate_gradient(sys, [3.123, 0.412212])


@code_llvm evaluate(sys, [3, 4])

v = [3.12321, -0.132]

@code_llvm evaluate(sys, v)
@code_native evaluate(sys, v)
@btime evaluate($sys, $v)

@btime evaluate_gradient($sys, $v)










f = ((2x+3y)^2 + 3)^3
g = ((2x+3y)^2 + 3)^2 + 4
f_x = differentiate(f, x)
f_y = differentiate(f, y)
f_y2 = differentiate(f_y, y)
f_yx = differentiate(f_y, x)

hash(f)

hash(x + 3)
hash((x + 2) + 1)

hash(f+g)

list = InstructionList()
push!(list, f)
push!(list, f_y)
push!(list, f_y2)
push!(list, f_yx)


push!(list, f_y2)
push!(list, f_y3)
list



















push!(list, g)
list

g = let
    g = f
    for i in 1:1
        g = differentiate(g, y)
    end
    g
end

HCModelKit.simplify(3(x + y) + 5(x+y))
a = 3(x + y) + 5(x+y)
u, v = a.args
u == Operation &&
u.func == :* &&
       u.args[1] isa Constant &&
       v == Operation &&
       v.func == :* &&
       v.args[1] isa Constant &&

list = InstructionList()
push!(list, f_x)
push!(list, f_y)
list















id_f = Base.push!(list, f)
id_fx = Base.push!(list, f_x)
id_fy = Base.push!(list, f_y)
id_fy2 = Base.push!(list, f_y2)


list


HCModelKit.derivative(list, [x,y, z])


struct ExpressionForest{N}
    list::InstructionList
    output::AbstractArray
end



Array{Int,0}(1, 2)



?Array



list.instructions.keys
list.instructions.vals

# 0d, 1d, 2d, ...

@code_warntype HCModelKit.derivative(list, [x,y, z])


dlist, _ = HCModelKit.derivative(list, [x,y, z])

dlist.instructions.vals












dlist.instructions.keys

@time Dict(id => instr for (instr, id) in list.instructions)



instrs = Instruction[]
instr_ids = HCModelKit.InstructionId[]
for (instr, id) in list.instructions
    push!(instrs, instr)
    push!(instr_ids, id)
end
instrs
instr_ids

instr_to_id = Dict(zip(instrs, 1:4))
id_to_instr = Dict(zip(instr_ids, 1:4))


instrs[4].args






























instrs = collect(list.instructions)




list.derivatives

id = HCModelKit.InstructionId(:x, 3)
HCModelKit.InstructionId(:x, 3) == HCModelKit.InstructionId(:x, 3)


import DiffRules

DiffRules.diffrule(:Base, :/, :x, :y)


#
#
# @which [2,3, 3,5]
# @which [2 3]
# @which [2; 3]
#
#
# f(n) = [i for i in 1:n]
# @code_lowered f
#
# @which collect(i for i in 1:3)
