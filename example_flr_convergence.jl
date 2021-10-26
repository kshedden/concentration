using Distributions, PyPlot, Printf, Statistics, UnicodePlots, Interpolations

include("mediation.jl")
include("functional_lr.jl")


n = 1000
m = 11
pg = collect(range(1 / m, 1 - 1 / m, length = m))

Random.seed!(1000)

age1 = 0.1 * randn(n)
age2 = 0.1 * randn(n)

Random.seed!(50)

# Generate the exposure
exs = 1.0
ex = exs .* randn(n)

# Generate the first mediator
m1a, m1b = 3.0, 4.0
m1 = m1a .* ex + m1b .* randn(n)
m1s = sqrt(m1a^2 * exs^2 + m1b^2)

# Generate the second mediator
m2a, m2b = 4.0, 3.0
m2 = m2a .* ex + m2b .* randn(n)
m2s = sqrt(m2a^2 * exs^2 + m2b^2)

# Generate the response
yb, y1b, y2b, ys = 3.0, 2.0, 2.0, 4.0
yv = yb .* ex + y1b .* m1 + y2b .* m2 + ys .* randn(n)

# Put everything into a design matrix
xm = hcat(age1, age2, ex, m1, m2)

# Center xm
if false
    xxm = xm[:, 1:4]
    xxm = Array{Float64,2}(xxm)

    xmn = mean(xxm, dims = 1)
    xsd = std(xxm, dims = 1)
    for j = 1:size(xxm, 2)
        xm[:, j] = (xm[:, j] .- xmn[j]) ./ xsd[j]
    end
end


# Marginal quantiles of the exposures and mediators
exq = quantile(Normal(0, exs), pg)
m1q = quantile(Normal(0, m1s), pg)
m2q = quantile(Normal(0, m2s), pg)

bw = [1.0, 1, 1, 1, 1]*0.25

# Compute conditional quantiles of y given x, m1, and m2.
xr = mediation_quantiles(yv, xm, pg, exq, m1q, m2q, bw)

# Remove the median
xc, md = center(xr)

cu = 0 .* fill(1.0, 3)
cv = 0 .* fill(1.0, 3)
r = length(size(xc))
p = size(xc)[1]
vx = vec(xc)
f, g! = _flr_fungrad_tensor(vx, p, r, cu, cv)

# Check the derivatives at this point
pa = vcat(vec(u), vec(v))
u, v = get_start(xc)

# Calculate the derivative analytically
grad1 = zeros(length(pa))
g!(grad1, pa, project=false)

# Calculate the derivative numerically
grad2 = zeros(length(pa))
for i in eachindex(pa)
	ee = 1e-5
	e = zeros(length(pa))
	e[i] = ee
	grad2[i] = (f(pa + e) - f(pa)) ./ ee
end

error("")

# Fit a low rank model to the estimated quantiles
u, v = fit_flr_tensor(xc, fill(1.0, 3), fill(1.0, 3))
