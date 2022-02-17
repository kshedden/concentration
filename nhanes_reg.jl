using DataFrames, CSV, Printf, UnicodePlots, LinearAlgebra
using Distributions, PyPlot

rm("plots", recursive = true, force = true)
mkdir("plots")

include("qreg_nn.jl")
include("flr_reg.jl")
include("basis.jl")
include("nhanes_prep.jl")

ifig = 0

sex = 2
da = select_sex(sex)

# Set up and run the quantile regression
y = Vector{Float64}(da[:, :BPXSY1])
X0 = Matrix{Float64}(da[:, [:RIDAGEYR_z, :BMXBMI_z, :BMXHT_z]])
nn = qreg_nn(y, X0)
yq = zeros(length(y), 9)
yqm = zeros(9)
ppy = collect(range(0.1, 0.9, length = 9))
for j = 1:9
    yq[:, j] = fit(nn, ppy[j], 0.1)
    yqm[j] = mean(yq[:, j])
    yq[:, j] .-= yqm[j]
end

pp = collect(range(0.01, 0.99, length = 101))
qq = quantile(Normal(0, 1), pp)

# Create basis functions for the low-rank model.
X, Xp = Vector{Matrix{Float64}}(), Vector{Matrix{Float64}}()
gr = collect(range(-2, 2, length = 101))
grx = []
gl = []
for x0 in eachcol(X0)
    B, g = genbasis(x0, 5, std(x0) / 2, linear = true)
    push!(gl, g)
    push!(X, B)
    push!(Xp, g(gr))
    push!(grx, gr)
end

# Plot the basis functions
function plot_basis(ifig)
    xp = first(Xp)
    PyPlot.clf()
    PyPlot.grid(true)
    for k = 1:size(xp, 2)
        PyPlot.plot(gr, xp[:, k])
    end
    PyPlot.savefig(@sprintf("plots/%03d.pdf", ifig))
    return ifig + 1
end
ifig = plot_basis(ifig)

# Fit the low rank model
cu, cv = 10, 1000
q = length(X)
fr = fitlr(X, Xp, yq, cu * ones(q), cv * ones(q))

# Check the explained variance
fv, _ = getfit(X, fr)
resid = yq - fv
for j = 1:size(yq, 2)
    println(@sprintf("%f %f", std(yq[:, j]), std(resid[:, j])))
end

function plots1(ifig)
    for k in eachindex(X)

        u = Xp[k] * fr.beta[k]
        m = u * fr.v[:, k]'

        PyPlot.clf()
        PyPlot.grid(true)
        PyPlot.plot(grx[k], u, "-")
        PyPlot.xlabel(@sprintf("%s Z-score", vnames[k]), size = 15)
        PyPlot.ylabel(@sprintf("%s PC score", vnames[k]), size = 15)
        PyPlot.savefig(@sprintf("plots/%03d.pdf", ifig))
        ifig += 1

        PyPlot.clf()
        PyPlot.grid(true)
        PyPlot.plot(ppy, fr.v[:, k], "-")
        if minimum(fr.v[:, k]) > 0
            PyPlot.ylim(bottom = 0)
        end
        if maximum(fr.v[:, k]) < 0
            PyPlot.ylim(top = 0)
        end
        PyPlot.xlabel("SBP probability points", size = 15)
        PyPlot.ylabel(@sprintf("%s loading", vnames[k]), size = 15)
        PyPlot.savefig(@sprintf("plots/%03d.pdf", ifig))
        ifig += 1

        mx = maximum(abs, m)
        PyPlot.clf()
        im = PyPlot.imshow(
            m,
            interpolation = "nearest",
            aspect = "auto",
            origin = "lower",
            extent = [minimum(ppy), maximum(ppy), minimum(grx[k]), maximum(grx[k])],
            cmap = "bwr",
            vmin = -mx,
            vmax = mx,
        )
        PyPlot.colorbar()
        PyPlot.xlabel("SBP quantiles", size = 15)
        PyPlot.ylabel(@sprintf("%s Z-score", vnames[k]), size = 15)
        PyPlot.savefig(@sprintf("plots/%03d.pdf", ifig))
        ifig += 1
    end
    return ifig
end

ifig = plots1(ifig)

f = [@sprintf("plots/%03d.pdf", j) for j = 0:ifig-1]
c = `gs -sDEVICE=pdfwrite -dNOPAUSE -dBATCH -dSAFER -sOutputFile=writing/nhanes_reg.pdf $f`
run(c)
