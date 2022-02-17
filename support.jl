#=
Find support points that represent a distribution.

Given a collection Y of vectors, find a set of support 
points of a specified size that capture the distribution
represented by Y.

The algorithm implemented below is based on equation 22 of
the following reference:

https://arxiv.org/abs/1609.01811
=#

using Random, PyPlot, LinearAlgebra

# The target function to be minimized, not explicitly used in the
# optimization algorithm.
function target(Y::Vector{T}, X::Vector{T}) where {T<:AbstractVector}

	n = length(X)
	N = length(Y)

	f = 0.0
	for y in Y
		for x in X
			f += norm(x - y)
		end
	end
	f *= 2 / (n * N)

	for j1 in eachindex(X)
		for j2 in 1:j1-1
			f -= 2 * norm(X[j1] - X[j2]) / n^2
		end
	end

	return f
end

# One iteration of fitting.  Returns an updated set of support points
# based on the current support points in X and the data in Y.
function _update_support(Y::Vector{T}, X::Vector{T}) where {T<:AbstractVector}

    # Size of the sample data.
    N = length(Y)

    # Dimension of the vectors
    d = length(first(Y))

    # Number of support points
    n = length(X)

    # Storage for the new support points
    X1 = Vector{T}()

    # Update each support point in turn
    for (i, xi) in enumerate(X)

        s = zeros(d)
        for (j, xj) in enumerate(X)
            if j != i
                u = xi - xj
                s += u / norm(u)
            end
        end
        s *= N / n

        q = 0.0
        for y in Y
            nm = norm(y - xi)
            s += y / nm
            q += 1 / nm
        end

        push!(X1, s / q)
    end

    return X1
end

#=
Find a set of 'n' support points that represent the distribution of
the values in 'Y'.
=#
function support(
    Y::Vector{T},
    n::Int;
    maxit = 1000,
    tol = 1e-4,
)::Vector{T} where {T<:AbstractVector}

    N = length(Y)

	# Starting values are a random sample from the data.
	# Need to perturb since the algorithm has a singularity
	# when a support point is exactly equal to a data point.
    ii = randperm(N)[1:n]
    d = length(first(Y))
    X = [copy(Y[i]) + 0.1 * randn(d) for i in ii]

    success = false
    for itr = 1:maxit
        X1 = _update_support(Y, X)

        # Assess convergence based on the L2 distance from the
        # previous support points to the current ones.
        di = 0.0
        for j in eachindex(X)
            di += norm(X1[j] - X[j])^2
        end
        di = sqrt(di)
        if di < tol
            success = true
            break
        end

        X = X1
    end

    if !success
        @warn "Support point estimation did not converge"
    end

    return X
end


function test1()

    n = 1000
    Y = Vector{Vector{Float64}}()
    for i = 1:n
        x = 2 * pi * rand()
        y = sin(x) + 0.3 * randn()
        push!(Y, [x, y])
    end

    X = support(Y, 10)

    PyPlot.clf()
    PyPlot.grid(true)
    x = [z[1] for z in Y]
    y = [z[2] for z in Y]
    PyPlot.plot(x, y, "o", mfc = "none")
    x = [z[1] for z in X]
    y = [z[2] for z in X]
    PyPlot.plot(x, y, "o", ms = 5, color = "red")
    PyPlot.savefig("test1.pdf")
end
