using Statistics, Distributions, Printf, LinearAlgebra

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
beta_mean = [0.0, 1, 0]
beta_var = [1.0, 0, 1]

# Determines the variance structure
het = 1.0
sigma = 1.0

# Algorithm parameters
nrep = 10
la = 0.1

# Work with quantiles at these probabiity points
pp = range(0.1, 0.9, length = 9)

# The mean and standard deviation for each observation
# in the simulated population
mu = xmat * beta_mean
sd = sigma * sqrt.(1 .+ het * clamp.(2 .+ xmat * beta_var, 0, Inf))

# The true quantiles
tq = zeros(n, length(pp))
for i = 1:n
    for j in eachindex(pp)
        tq[i, j] = quantile(Normal(mu[i], sd[i]), pp[j])
    end
end

# Find the true directions that capture all mean and
# variance structure.  Based on the way the data are
# simulated, there are only two non-null factors.
tqc = copy(tq)
center!(tqc)
u2, s2, v2 = svd(tqc)

function simstudy()

	# Generate data
	y = mu + sd .* randn(n)

	npc = 2
	nperm = 0
	eta, beta, qhc, xmat1, ss, sp = qnn_cca(y, xmat, npc, nperm)

	# Calculate how accurately we recovered the quantile profiles.
	a1 = canonical_angles(qhc * eta[:, 1:2], qhc * v2[:, 1:2])

	# Calculate how accurately we recovered the variable profiles.
	tb = hcat(beta_mean, beta_var)
	a2 = canonical_angles(xmat1 * beta[:, 1:2], xmat * tb)

	return acos.(clamp.(a1, -1, 1)), acos.(clamp.(a2, -1, 1))
end

nrep = 100
eta1_ang = zeros(nrep, 2)
beta1_ang = zeros(nrep, 2)
for itr in 1:nrep
	e, b = simstudy()
	eta1_ang[itr, :] = e
	beta1_ang[itr, :] = b
end

# Not used now
# Matrix that maps solution to population:
# eta -> eta * F
# beta1 -> beta1 * F
F = (beta' * xmat' * xmat * beta) \ (beta' * xmat' * xmat * tb)
