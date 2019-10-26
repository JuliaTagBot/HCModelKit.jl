using Pkg;
Pkg.activate(@__DIR__)

using HCModelKit


x = Variable(:x)
y = Variable(:y)

a = 3 + x

list = InstructionList(x*y)

instrs = collect(list.instructions)

HCModelKit.derivative!(list, instrs[1]...)
list

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
