using HCModelKit
using Test

@testset "HCModelKit.jl" begin
    x = Variable("x")

    e1 = convert(Expression, :(x^3+x^2))
    e2 = x^3 + x^2
    @test e1 == e2

    e3 = convert(Expression, :(sin(x^3+3)))
    e4 = sin(x^3+3)
    @test e3 == e4
    @test e4.func == (:Base, :sin)
end
