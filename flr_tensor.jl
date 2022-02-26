using LinearAlgebra, IterTools

#=
Takes a tensor x and returns design matrices X, Xp and a response
matrix Y that can be passed to fitlr (in flr_reg.jl).
=#
function setup_tensor(x::AbstractArray)

    sx = size(x)
    @assert minimum(sx) == maximum(sx) # All axes must have the same length
    r = length(sx)
    p = first(sx)
    p2 = div(p, 2) + 1 # The mid-point, prefer to use odd p
    q = r - 1
    n = prod(sx[1:end-1])
    Y = zeros(n, p)

    X, Xp = Vector{Matrix{Float64}}(), Vector{Matrix{Float64}}()
    for j = 1:q
        xm = zeros(n, p - 1)
        push!(X, xm)
        ip = I(p)[:, [j for j in 1:p if j != p2]]
        push!(Xp, ip)
    end

    i = 1
    for ii in product(Iterators.repeated(1:p, r)...)
        j = (i - 1) % n + 1
        for k = 1:q
            if ii[k] != p2
                l = ii[k] < p2 ? ii[k] : ii[k] - 1
                X[k][j, l] = 1
            end
        end
        Y[j, ii[end]] = x[ii...]
        i += 1
    end

    return (X, Xp, Y)
end

#=
Construct a fitted tensor from the parameters in u, v
and ca (the central axis).
=#
function getfit_tensor(
    u::AbstractMatrix{Float64},
    v::AbstractMatrix{Float64},
    ca::AbstractVector{Float64},
)::Array{Float64}

    @assert size(u, 1) == size(v, 1) == length(ca)
    @assert size(u, 2) == size(v, 2)
    p, q = size(u)
    r = q + 1

    di = fill(p, r)
    x = zeros(di...)

    for ii in product(Iterators.repeated(1:p, r)...)
        x[ii...] = ca[ii[end]]
        for j = 1:q
            x[ii...] += u[ii[j], j] * v[ii[end], j]
        end
    end

    return x
end

#=
Takes a tensor x and removes its central axis.  The
centered tensor and central axis are returned.
=#
function center_tensor(x::Array{Float64})
    sx = size(x)
    @assert minimum(sx) == maximum(sx)
    r = length(sx)
    p = first(sx)
    p2 = div(p, 2) + 1
    ca = zeros(p)
    ii = fill(p2, r)
    for j = 1:p
        ii[end] = j
        ca[j] = x[ii...]
    end
    xx = copy(x)
    for ii in product(Iterators.repeated(1:p, r)...)
        xx[ii...] = x[ii...] - ca[ii[end]]
    end
    return (xx, ca)
end
