using Optim, LinearAlgebra, IterTools, Statistics, Random
using UnicodePlots # for testing only

# Construct a smoothing penalty matrix for a vector of length p.
# Each row contains the discreteized second derivative at a 
# specific position.
function _fpen(p)
    a = zeros(p-2, p)
    for i in 1:p-2
        a[i, i:i+2] = [1 -2 1]
    end
    return a
end

# Split the parameter vector for the additive low-rank tensor model into components.
# If there are q covariates and the grid for each has length p, then the data is a
# tensor with r = q + 1 axes and p points along each axis.  The _split function
# reshapes the parameters into two p x q arrays, u and v, that are views
# into the input argument z.
function _split(z, p, q)
    @assert(length(z) % 2 == 0)
    m = div(length(z), 2)
    @assert(m % p == 0)
    u = reshape(@view(z[1:m]), p, q)
    v = reshape(@view(z[m+1:end]), p, q)
    return tuple(u, v)
end

# Return the objective function and corresponding gradient functions
# that are used to fit the low rank model.
function _flr_fungrad_tensor(x::Array{Float64,1}, p::Int, r::Int, cu::Array{Float64,1}, cv::Array{Float64,1})

    @assert length(x) == p^r

    # The number of covariates
    q = r - 1

    # The size of the tensor representation of the data along each axis
	di = fill(p, r)

    # Convert the data to a tensor
    xr = reshape(x, di...)

    # Storage
    ft = zeros(di...)
    rs = zeros(di...)

    # The smoothing penalty matrix.
    w = _fpen(p)
    w2 = w' * w

    # Set the fitted values and residuals
    function _getfit(u, v)
        ft .= 0 # Fitted values
        for ii in product(Iterators.repeated(1:p, r)...)
            for k in 1:q
                ft[ii...] += u[ii[k], k] * v[ii[end], k]
            end
        end
        rs .= xr .- ft # Residuals
    end

    # The fitting function (sum of squared residuals).
    f = function(z::Array{Float64,1})

        u, v = _split(z, p, q)
        _getfit(u, v)

        # Sum of squared residuals
        rv = sum(abs2, rs)

        # Penalty terms
        for j in 1:q
            rv += cu[j] * sum(abs2, w*u[:,j]) / sum(abs2, u[:,j])
            rv += cv[j] * sum(abs2, w*v[:,j]) / sum(abs2, v[:,j])
        end

        return rv
    end

    # The gradient of the fitting function
    g! = function(G, z; project=true)

        # Split the parameters
        u, v = _split(z, p, q)

        _getfit(u, v)

        # Sum of squared residuals
        rv = sum(abs2, rs)

        # Split the gradient
        G .= 0
        ug, vg = _split(G, p, q)

        # Penalty gradients
        for j in 1:q
            ssu = sum(abs2, u[:,j])
            nu = sum(abs2, w*u[:,j])
            ssv = sum(abs2, v[:,j])
            nv = sum(abs2, w*v[:,j])
            ug[:,j] = 2 .* cu[j] .* (ssu .* w2 * u[:,j] - nu .* u[:,j]) ./ ssu^2
            vg[:,j] = 2 .* cv[j] .* (ssv .* w2 * v[:,j] - nv .* v[:,j]) ./ ssv^2
        end

        # Loss gradient
        for ii in product(Iterators.repeated(1:p, r)...)
            f = -2 * rs[ii...]
            for j in 1:q
                ug[ii[j], j] += f * v[ii[end], j]
                vg[ii[end], j] += f * u[ii[j], j]
            end
        end

        # The problem has two constraints: each column of u is normalized to unit
        # length, and all but the first column of u are centered.  If projecting,
        # the linearized versions of these constraints are removed from the gradient.
        if project

            proj = zeros(p, 2)
            proj[:, 1] .= 1

            for j in 1:q

                if j > 1
                    # Remove two constraints
                    proj[:, 2] = u[:, j]
                    u_, s_, v_ = svd(proj)
                    ug[:, j] .-= u_ * (u_' * ug[:, j])
                else
                    # Remove one constraint
                    f = sum(abs2, u[:, j])
                    ug[:, j] .-= dot(ug[:, j], u[:, j]) * u[:, j] / f
                end

            end
        end
    end

    return tuple(f, g!)

end

# Use a series of SVD's to obtain starting values for the model parameters.
# To obtain starting values for column j of u and v, the tensor is averaged
# over all axes except axis j and the final axis, then the dominant singular
# vectors are computed.
function get_start(x::Array{Float64,1}, p::Int, r::Int)

    x = copy(x)

    q = r - 1
    u = zeros(p, q)
    v = zeros(p, q)
    z = zeros(p, p)

    di = p * ones(Int, r)
    xr = reshape(x, di...)

    for j in 1:q

        # Average the tensor over all axes except j and the final axis.
        z .= 0
        for ii in product(Iterators.repeated(1:p, r)...)
            z[ii[j], ii[end]] += xr[ii...]
        end
        z .= z ./ p^(q-1)

        # Except for the first term, the u vector should be centered.
        if j > 1
            for j in 1:p
                z[:, j] .-= mean(z[:, j])
            end
        end

        uu, ss, vv = svd(z)
        f = sqrt(ss[1])
        u[:, j] = f * uu[:, 1]
        v[:, j] = f * vv[:, 1]

        # Use this scaling to make the representation unique.
        f = norm(u[:, j])
        u[:, j] /= f
        v[:, j] *= f

    end

    return u, v

end

# Fit a functional low-rank model to the tensor x.  The input data x is a flattened representation
# of a tensor with r axes, each having length p.
function fit_flr_tensor(x::Array{Float64,1}, p::Int, r::Int, cu::Array{Float64,1}, cv::Array{Float64,1}; start=nothing)

    # Number of covariates
    q = r - 1

    # Starting values
    if !isnothing(start)
        @assert length(start) == 2 * p * q
        u, v = _split(start, p, q)
    else
        u, v = get_start(x, p, r)
    end

    pa = vcat(u[:], v[:])

    f, g! = _flr_fungrad_tensor(x, p, r, cu, cv)

    r = optimize(f, g!, pa, LBFGS(), Optim.Options(iterations=50, show_trace=true))
    println(r)

    if !Optim.converged(r)
        println("fit_flr_tensor did not converge")
    end

    z = Optim.minimizer(r)
    uh, vh = _split(z, p, q)
    return tuple(uh, vh)

end


# Reparametrize the model from the form that is directly
# fit to the data, to a form that is more interpretable.
# The form that is directly fit to the data is u*v'.
# The form produced here takes the fitted value
# at the central position of u, denoted 'mu', and 
# adjusts u accordingly to produce new scores uu.
# The fitted values are identical: u*v' = mu + uu*v'.
# Note that v is not changed by the reparameterization.
function reparameterize(u, v)

    @assert size(u, 1) == size(v, 1)
    @assert size(u, 2) == size(v, 2)
    p, q = size(u)
    u = copy(u)
    v = copy(v)

	# The central position of the u axis.
    m = div(p, 2)

    # The central value that is split out from
    # the factor representation.
    mu = v * u[m, :]
    
    for j in 1:q
        u[:, j] .-= u[m, j]
    end

    return tuple(mu, u, v)

end
