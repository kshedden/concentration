function genbasis(x, b, s; linear = false)

    fl = []
    B = Vector{Vector{Float64}}()

    if linear
        f0 = x -> x
        push!(fl, f0)
        push!(B, f0(x))
    end

    # Basis centers
    c = if b == 1
        [mean(x)]
    else
        range(minimum(x), maximum(x), length = b)
    end

    for j = 1:b
        y0 = (x .- c[j]) / s
        y0 = exp.(-y0 .^ 2 / 2)
        f = function (x)
            y = (x .- c[j]) / s
            return exp.(-y .^ 2 / 2) .- exp(-c[j]^2 / (2 * s^2))
        end
        push!(B, f(x))
        push!(fl, f)
    end

    function g(z)
        x = zeros(length(z), length(fl))
        for j in eachindex(fl)
            x[:, j] = fl[j](z)
        end
        return x
    end

    return hcat(B...), g
end
