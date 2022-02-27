using Statistics, Distributions, Printf, LinearAlgebra, DataFrames

include("qreg_nn.jl")
include("cancorr.jl")

# Sample size
n = 1500

# Generate correlated covariates.
r = 0.5
cm = r * ones(3, 3) + (1 - r) * I(3)
cr = cholesky(cm)
xmat = randn(n, 3) * cr.U
center!(xmat)

# Linear predictors for the mean and variance structure.
beta_mean = Float64[0, 1, 0]
beta_var = Float64[1, 0, 1]

# Determines the variance structure
het = 1.0

# Algorithm parameters
nrep = 100
la = 0.1

# Work with quantiles at these probabiity points
pp = range(0.1, 0.9, length = 9)

# The mean and standard deviation for each observation
# in the simulated population
mu = xmat * beta_mean

function get_true_quantiles(sd)
    # The true quantiles
    tq = zeros(n, length(pp))
    for i = 1:n
        for j in eachindex(pp)
            tq[i, j] = quantile(Normal(mu[i], sd[i]), pp[j])
        end
    end

    # Find a basis for the subspace that captures all mean 
    # and variance structure.  Based on the way the data are
    # simulated, there are only two non-null factors.
    tqc = copy(tq)
    center!(tqc)
    _, s2, v2 = svd(tqc)
	@assert sum(abs.(s2) .> 1e-8) == 2

    return tq, v2
end

function simstudy(npc, sigma)

    # The conditional standard deviation
    sd = sigma * sqrt.(1 .+ het * clamp.(2 .+ xmat * beta_var, 0, Inf))

    tq, v2 = get_true_quantiles(sd)

    # Generate data
    y = mu + sd .* randn(n)

    # We are only considering the estimates here so no need to do
    # permutations
    nperm = 0

    eta, beta, qhc, xmat1, ss, sp = qnn_cca(y, xmat, npc, nperm)

    # Calculate how accurately we recovered the quantile profiles.
    a1 = canonical_angles(qhc * eta[:, 1:2], qhc * v2[:, 1:2])

    # Calculate how accurately we recovered the variable profiles.
    tb = hcat(beta_mean, beta_var)
    a2 = canonical_angles(xmat1 * beta[:, 1:2], xmat * tb)

    return a1, a2
end

# A place to put the summarized simulation results.
rslt = DataFrame(
    :npc => Int[],
    :sigma => Float64[],
    :beta_ang_mean => Float64[],
    :beta_ang_sd => Float64[],
    :eta1_ang_mean => Float64[],
    :eta1_ang_sd => Float64[],
    :eta2_ang_mean => Float64[],
    :eta2_ang_sd => Float64[],
)

for sigma in Float64[1, 2]
    for npc in [3, 4, 5]
        eta1_ang = zeros(nrep, 2)
        beta1_ang = zeros(nrep, 2)
        for itr = 1:nrep
            e, b = simstudy(npc, sigma)
            eta1_ang[itr, :] = e
            beta1_ang[itr, :] = b
        end

        # The first canonical angle for beta is always zero (X-side)
        b_mean = mean(beta1_ang[:, 2])
        b_sd = std(beta1_ang[:, 2])
        row = [npc, sigma, b_mean, b_sd]

        # The first and second canonical angles for eta (Y-side)
        push!(row, mean(eta1_ang[:, 1]), std(eta1_ang[:, 1]))
        push!(row, mean(eta1_ang[:, 2]), std(eta1_ang[:, 2]))
        push!(rslt, row)
    end
end
