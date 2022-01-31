using Optim, LinearAlgebra, IterTools, Statistics, Random
using UnicodePlots # for testing only

# Construct a smoothing penalty matrix for a vector of length p.
# Each row contains the discreteized second derivative at a 
# specific position.
function _fpen(p)
    a = zeros(p - 2, p)
    for i = 1:p-2
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

# Get the fitted values and residuals for the low rank
# tenor model represented by (u, v).  xr are the observed
# values used to form the residuals.  The fitted values
# are written into ft and the residuals are written into
# rs.
function getfitresid(
    u::AbstractArray,
    v::AbstractArray,
    xr::AbstractArray,
    ft::AbstractArray,
    rs::AbstractArray,
)
    p = size(u, 1)
    q = size(u, 2)
    r = q + 1

    ft .= 0 # Fitted values
    for ii in product(Iterators.repeated(1:p, r)...)
        for k = 1:q
            ft[ii...] += u[ii[k], k] * v[ii[end], k]
        end
    end
    # Calculate residuals only if storage is provided
    if length(xr) > 0
        rs .= xr .- ft
    end
end

# Get the fitted values and residuals without pre-allocated storage.
function getfitresid(u::AbstractArray, v::AbstractArray, xr::AbstractArray)
    p = size(u, 1)
    q = size(u, 2)
    r = q + 1
    di = fill(p, r)
    ft = zeros(di...)
    rs = zeros(di...)
    getfitresid(u, v, xr, ft, rs)
    return (ft, rs)
end

# Get the fitted values from a low-rank model specified by u, v, with dimensions p, r.
function getfit(u::AbstractArray, v::AbstractArray)
    p = size(u, 1)
    q = size(u, 2)
    r = q + 1
    di = fill(p, r)
    xr = zeros(di...)
    ft = zeros(di...)
    rs = zeros(di...)
    getfitresid(u, v, xr, ft, rs)
    return ft
end

# Return the objective function and corresponding gradient functions
# that are used to fit the low rank model.  The data 'x' are a 
# vectorized tensor with 'r' axes and 'p' values along each axis. 
function _flr_fungrad_tensor(
    x::Vector{Float64},
    p::Int,
    r::Int,
    cu::Vector{Float64},
    cv::Vector{Float64},
)
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

    # The fitting function (sum of squared residuals).
    f = function (z::Vector{Float64})

        u, v = _split(z, p, q)
        getfitresid(u, v, xr, ft, rs)

        # Sum of squared residuals
        rv = sum(abs2, rs)

        # Penalty terms
        for j = 1:q
            # DEBUG
            #rv += cu[j] * sum(abs2, w * u[:, j]) / sum(abs2, u[:, j])
            #rv += cv[j] * sum(abs2, w * v[:, j]) / sum(abs2, v[:, j])
            rv += cu[j] * sum(abs2, w * u[:, j])
            rv += cv[j] * sum(abs2, w * v[:, j])
        end

        return rv
    end

    # The gradient of the fitting function
    g! = function (G, z; project = true)

        # Split the parameters
        u, v = _split(z, p, q)
        getfitresid(u, v, xr, ft, rs)

        # Sum of squared residuals
        rv = sum(abs2, rs)

        # Split the gradient
        G .= 0
        ug, vg = _split(G, p, q)

        # Penalty gradients
        for j = 1:q
            # DEBUG
            #ssu = sum(abs2, u[:, j])
            #nu = sum(abs2, w * u[:, j])
            #ssv = sum(abs2, v[:, j])
            #nv = sum(abs2, w * v[:, j])
            #ug[:, j] = 2 .* cu[j] .* (ssu .* w2 * u[:, j] - nu .* u[:, j]) ./ ssu^2
            #vg[:, j] = 2 .* cv[j] .* (ssv .* w2 * v[:, j] - nv .* v[:, j]) ./ ssv^2
            ug[:, j] = 2 .* cu[j] .* w2 * u[:, j]
            vg[:, j] = 2 .* cv[j] .* w2 * v[:, j]
        end

        # Loss gradient
        for ii in product(Iterators.repeated(1:p, r)...)
            f = -2 * rs[ii...]
            for j = 1:q
                ug[ii[j], j] += f * v[ii[end], j]
                vg[ii[end], j] += f * u[ii[j], j]
            end
        end

        # Project so that each column of the gradient of u is centered
        if project
            for j = 1:size(ug, 2)
                ug[:, j] .-= mean(ug[:, j])
            end
        end
    end

    hess! = function (H, z; project = true)

        u, v = _split(z, p, q)
        getfitresid(u, v, xr, ft, rs)
        H .= 0

        for ii in product(Iterators.repeated(1:p, r)...)

            # Update the u x u Hessian terms for the loss function
            for k1 = 1:q
                for k2 = 1:q
                    j1 = (k1 - 1) * p + ii[k1]
                    j2 = (k2 - 1) * p + ii[k2]
                    H[j1, j2] += 2 * v[ii[end], k1] * v[ii[end], k2]
                end
            end

            # Update the v x v Hessian terms for the loss function
            for k1 = 1:q
                for k2 = 1:q
                    j1 = p * q + (k1 - 1) * p + ii[end]
                    j2 = p * q + (k2 - 1) * p + ii[end]
                    H[j1, j2] += 2 * u[ii[k1], k1] * u[ii[k2], k2]
                end
            end

            # Update the u x v Hessian terms for the loss function
            for k1 = 1:q
                for k2 = 1:q
                    f = k1 == k2 ? rs[ii...] : 0
                    f -= u[ii[k2], k2] * v[ii[end], k1]
                    j1 = (k1 - 1) * p + ii[k1]
                    j2 = p * q + (k2 - 1) * p + ii[end]
                    H[j1, j2] -= 2 * f
                    H[j2, j1] -= 2 * f
                end
            end
        end

        # Update for the penalty function
        for j = 1:q
            j1 = (j - 1) * p
            H[j1+1:j1+p, j1+1:j1+p] += 2 * cu[j] * w2
            j1 += p * q
            H[j1+1:j1+p, j1+1:j1+p] += 2 * cv[j] * w2
        end

        if project
            oo = ones(p)
            ct = I(p) - oo * oo' / p
            for j1 = 1:q
                for j2 = 1:q
                    i1 = (j1 - 1) * p
                    i2 = (j2 - 1) * p
                    H[i1+1:i1+p, i2+1:i2+p] = ct * H[i1+1:i1+p, i2+1:i2+p] * ct
                end
            end
        end
        H .= (H + H') / 2
    end

    return tuple(f, g!, hess!)
end

# Use a series of SVD's to obtain starting values for the model parameters.
# To obtain starting values for column j of u and v, the tensor is averaged
# over all axes except axis j and the final axis, then the dominant singular
# vectors are computed.
function get_start(xr::AbstractArray)::Tuple{AbstractArray,AbstractArray}

    sx = size(xr)
    r = length(sx)
    @assert minimum(sx) == maximum(sx) # All axes must have the same length
    p = first(sx)
    q = r - 1
    u = zeros(p, q)
    v = zeros(p, q)
    z = zeros(p, p)
    di = fill(p, r)

    for j = 1:q

        # Average the tensor over all axes except j and the final axis.
        z .= 0
        for ii in product(Iterators.repeated(1:p, r)...)
            z[ii[j], ii[end]] += xr[ii...]
        end
        z ./= p^(q - 1)

        uu, ss, vv = svd(z)
        f = sqrt(ss[1])
        u[:, j] = f * uu[:, 1]
        v[:, j] = f * vv[:, 1]

        # Use this scaling to make the representation unique.
        f = norm(v[:, j])
        u[:, j] *= f
        v[:, j] /= f

    end

    for j = 1:size(u, 2)
        u[:, j] .-= mean(u[:, j])
    end

    return u, v
end

# Fit a functional low-rank model to the tensor x.  The input data x is a flattened representation
# of a tensor with r axes, each having length p.
function fit_flr_tensor(
    x::AbstractArray,
    cu::Vector{Float64},
    cv::Vector{Float64};
    start = nothing,
)
    r = length(size(x))
    p = size(x)[1]
    vx = vec(x)

    # Number of covariates
    q = r - 1

    # Starting values
    if !isnothing(start)
        @assert length(start) == 2 * p * q
        u, v = _split(start, p, q)
    else
        u, v = get_start(x)
        u[div(p, 2)+1, :] .= 0 # Assume that central axis is removed
    end

    pa = vcat(u[:], v[:])

    f, g!, hess! = _flr_fungrad_tensor(vx, p, r, cu, cv)

    # Calculate the norm of the Newton step hess^-1 * grad
    # Not currently used
    hessigrad = function (pa)
        grad = zeros(length(pa))
        g!(grad, pa; project = false)
        H = zeros(length(pa), length(pa))
        hess!(H, pa; project = false)
        println(f(pa), " ", norm(grad), " ", norm(pinv(H) * grad))
    end

    # Gradient descent
    opt = Optim.Options(iterations = 10, show_trace = true)
    r = optimize(
        f,
        g!,
        pa,
        GradientDescent(linesearch = Optim.LineSearches.BackTracking()),
        opt,
    )

    # Conjugate gradient
    pa = Optim.minimizer(r)
    opt = Optim.Options(iterations = 2*length(pa), show_trace = true)
    r = optimize(f, g!, pa, LBFGS(linesearch = Optim.LineSearches.BackTracking()), opt)
    pa = Optim.minimizer(r)
	hessigrad(pa)

    # Newton
    pax = copy(pa) # Save in case Newton fails
    hess = zeros(length(pa), length(pa))
    grad = zeros(length(pa))
    success = false
    f0, f1 = f(pa), nothing
    println(f0)
    for k = 1:100
        g!(grad, pa; project = true)
        hess!(hess, pa; project = true)
        s,_ = eigen(hess)
        delta = pinv(hess) * grad # The Newton step
        pa .-= delta
        f1 = f(pa)
        println(f1, " ", norm(grad), " ", norm(delta), " ",
                sum(s .< 0))
        ndelta = norm(delta)
        if ndelta < 1e-7
            success = true
            break
        end
    end

    if f1 > f0
        # Newton went uphill, use CG solution
        println("warning: Newton optimization failed")
        pa = pax
    end

    if !success
        println("fit_flr_tensor did not converge")
    end

    z = Optim.minimizer(r)
    uh, vh = _split(z, p, q)

    # Normalize the representation.  There are various ways to do this,
    # but here we make the loading vectors have unit norm.
    for j = 1:size(vh, 2)
        f = norm(vh[:, j])
        vh[:, j] ./= f
        uh[:, j] .*= f
    end

    return tuple(uh, vh)
end

# Remove the "central axis" from the tensor x.  The central axis is
# x[d, d, ..., d, :] where d is the midpoint of the axis.
function center(x::AbstractArray)::Tuple{AbstractArray,AbstractArray}

    x = copy(x)
    sx = size(x)
    r = length(sx)
    m = sx[1] # All axes assumed to have the same length
    d = div(m, 2) + 1

    # Extract the central row (the row at the midpoint of
    # all axes except the last axis, and taking all elements of the last
    # axis).
    ii = fill(d, r)
    md = zeros(m)
    for i = 1:m
        ii[end] = i
        md[i] = x[ii...]
    end

    # Remove the central row
    for ii in product(Iterators.repeated(1:m, r)...)
        x[ii...] -= md[ii[end]]
    end

    return tuple(x, md)
end
