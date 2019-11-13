using HCModelKit
using Test

import FiniteDifferences
const FD = FiniteDifferences
using LinearAlgebra: det, ⋅

@testset "HCModelKit.jl" begin
    @testset "System" begin
        @var x y z
        function F_test(x)
            let (x, y, z) = (x[1], x[2], x[3])
                [
                 (x^2 + y + z + 2)^2 - 3,
                 4 * x^2 * z^2 * y + 4z - 6x * y * z^2,
                 (-z) / (x + y),
                ]
            end
        end

        F = CompiledSystem(System(F_test([x, y, z]), [x, y, z]))

        @test size(F) == (3, 3)
        @test length(F) == 3
        @test size(F, 1) == 3
        @test size(F, 2) == 3
        u = [0.0, 0.0, 0]
        U = zeros(3, 3)
        v = [0.4814, -0.433, -0.82709]

        fdm = FD.central_fdm(5, 1)
        û = F_test(v)
        @test evaluate(F, v) ≈ û
        @test evaluate!(u, F, v) ≈ û
        @test u ≈ û
        Û = FD.jacobian(fdm, F_test, v)[1]
        @test jacobian(F, v) ≈ Û
        @test jacobian!(U, F, v) ≈ Û
        @test U ≈ Û rtol = 1e-8
        u .= 0
        U .= 0
        @test evaluate_jacobian!(u, U, F, v) === nothing
        @test u ≈ û
        @test U ≈ Û
    end


    @testset "Homotopy" begin
        @var x y z t

        function H_test(x, s)
            let (x, y, z, t) = (x[1], x[2], x[3], s)
                [x^2 + y + z + 2t, 4 * x^2 * z^2 * y + 4z - 6x * y * z^2]
            end
        end

        H = CompiledHomotopy(Homotopy(H_test([x, y, z], t), [x, y, z], t))
        @test size(H) == (2, 3)
        @test size(H, 1) == 2
        @test size(H, 2) == 3
        @test length(H) == 2

        u = [0.0, 0.0]
        U = zeros(2, 3)
        v = [0.4814, -0.433, -0.82709]
        s = 0.359

        fdm = FD.central_fdm(5, 1)
        û = H_test(v, s)
        @test evaluate(H, v, s) ≈ û
        @test evaluate!(u, H, v, s) ≈ û
        @test u ≈ û
        Û = FD.jacobian(fdm, v -> H_test(v, s), v)[1]
        @test jacobian(H, v, s) ≈ Û
        @test jacobian!(U, H, v, s) ≈ Û
        @test U ≈ Û rtol = 1e-8
        u .= 0
        U .= 0
        @test evaluate_jacobian!(u, U, H, v, s) === nothing
        @test u ≈ û
        @test U ≈ Û

        û = FD.grad(fdm, s -> H_test(v, s), s)[1]
        @test dt(H, v, s) ≈ û
        @test dt!(u, H, v, s) ≈ û
        u .= 0
        U .= 0
        @test dt_jacobian!(u, U, H, v, s) === nothing
        @test u ≈ û
        @test U ≈ Û
    end

    @testset "Modeling" begin
        @testset "Bottleneck" begin
            @var x y z
            f = [(0.3 * x^2 + 0.5z + 0.3x + 1.2 * y^2 - 1.1)^2 +
                 (0.7 * (y - 0.5x)^2 + y + 1.2 * z^2 - 1)^2 - 0.3]

            I = let
                x = variables(f)
                n, m = length(x), length(f)
                @unique_var y[1:n] v[1:m] w[1:m]
                J = [differentiate(fᵢ, xᵢ) for fᵢ in f, xᵢ in x]
                f′ = [subs(fᵢ, x => y) for fᵢ in f]
                J′ = [subs(gᵢ, x => y) for gᵢ in J]
                Nx = (x - y) - J' * v
                Ny = (x - y) - J′' * w
                System([f; f′; Nx; Ny], [x; y; v; w])
            end
            @test I isa System
            @test size(I) == (8, 8)
        end

        @testset "Steiner" begin
            @var x[1:2] a[1:5] c[1:6] y[1:2, 1:5]

            #tangential conics
            f = a[1] * x[1]^2 + a[2] * x[1] * x[2] + a[3] * x[2]^2 + a[4] * x[1] +
                a[5] * x[2] + 1
            ∇ = differentiate(f, x)
            #5 conics
            g = c[1] * x[1]^2 + c[2] * x[1] * x[2] + c[3] * x[2]^2 + c[4] * x[1] +
                c[5] * x[2] + c[6]
            ∇_2 = differentiate(g, x)
            #the general system
            #f_a_0 is tangent to g_b₀ at x₀
            function Incidence(f, a₀, g, b₀, x₀)
                fᵢ = f(x => x₀, a => a₀)
                ∇ᵢ = [∇ᵢ(x => x₀, a => a₀) for ∇ᵢ in ∇]
                Cᵢ = g(x => x₀, c => b₀)
                ∇_Cᵢ = [∇ⱼ(x => x₀, c => b₀) for ∇ⱼ in ∇_2]

                [fᵢ; Cᵢ; det([∇ᵢ ∇_Cᵢ])]
            end
            @var v[1:6, 1:5]
            I = vcat(map(i -> Incidence(f, a, g, v[:, i], y[:, i]), 1:5)...)
            F = System(I, [a; vec(y)], vec(v))
            @test size(F) == (15, 15)
        end

        @testset "Reach plane curve" begin
            @var x y
            f = (x^3 - x*y^2 + y + 1)^2 * (x^2 + y^2 - 1) + y^2 - 5
            ∇ = differentiate(f, [x;y]) # the gradient
            H = differentiate(∇, [x;y]) # the Hessian

            g = ∇ ⋅ ∇
            v = [-∇[2]; ∇[1]]
            h = v' * H * v
            dg = differentiate(g, [x;y])
            dh = differentiate(h, [x;y])

            ∇σ = g .* dh - ((3/2) * h).* dg

            F = System([v ⋅ ∇σ; f], [x,y])
            @test size(F) == (2, 2)
        end
    end
end
