using LinearAlgebra, IterTools

function setup_tensor(x::AbstractArray)

    sx = size(x)
    @assert minimum(sx) == maximum(sx) # All axes must have the same length
    r = length(sx)
    p = first(sx)
    q = r - 1
    n = prod(sx[1:end-1])
    Y = zeros(n, p)

    X, Xp = Vector{Matrix{Float64}}(), Vector{Matrix{Float64}}()
    for j = 1:q
        xm = zeros(n, p)
        push!(X, xm)
        push!(Xp, I(p))
    end

    i = 1
    for ii in product(Iterators.repeated(1:p, r)...)
        j = (i - 1) % n + 1
        for k = 1:q
            X[k][j, ii[k]] = 1
        end
        Y[j, ii[end]] = x[ii...]
        i += 1
    end

    return (X, Xp, Y)
end
