using DataFrames, CSV, Printf, UnicodePlots, LinearAlgebra
using Distributions, PyPlot

rm("plots", recursive = true, force = true)
mkdir("plots")

include("qreg_nn.jl")
include("flr_reg.jl")
include("basis.jl")
include("nhanes_prep.jl")
include("plot_utils.jl")

ifig = 0

ppy = collect(range(0.1, 0.9, length = 9))

# Plot the basis functions
function plot_basis(Xp, gr, ifig)
    xp = first(Xp)
    PyPlot.clf()
    PyPlot.grid(true)
    for k = 1:size(xp, 2)
        PyPlot.plot(gr, xp[:, k])
    end
    PyPlot.savefig(@sprintf("plots/%03d.pdf", ifig))
    return ifig + 1
end

function runx(sex, ifig)

    sexs = sex == 2 ? "Female" : "Male"
    da = select_nhanes(sex, 18, 40)

    # Set up and run the quantile regression
    y = Vector{Float64}(da[:, :BPXSY1])
    X0 = Matrix{Float64}(da[:, [:RIDAGEYR_z, :BMXBMI_z, :BMXHT_z]])
    nn = qreg_nn(y, X0)
    yq = zeros(length(y), 9)
    yqm = zeros(9)
    for j = 1:9
        yq[:, j] = fit(nn, ppy[j], 0.1)
        yqm[j] = mean(yq[:, j])
        yq[:, j] .-= yqm[j]
    end

    # Create basis functions for the low-rank model.
    X, Xp = Vector{Matrix{Float64}}(), Vector{Matrix{Float64}}()
    gr = collect(range(-2, 2, length = 101))
    grx = []
    for x0 in eachcol(X0)
        B, g = genbasis(x0, 5, std(x0) / 2, linear = true)
        push!(X, B)
        push!(Xp, g(gr))
        push!(grx, gr)
    end

    if sex == 2
        ifig = plot_basis(Xp, gr, ifig)
    end

    # Fit the low rank model
    cu, cv = 10, 1000
    q = length(X)
    fr = fitlr(X, Xp, yq, cu * ones(q), cv * ones(q))

    # Check the explained variance
    fv, _ = getfit(X, fr)
    resid = yq - fv
    println("$(sexs) explained variances:")
    for j = 1:size(yq, 2)
        println(@sprintf("%f %f", std(yq[:, j]), std(resid[:, j])))
    end
    println("")

    ifig = plots_flr(sex, X, Xp, ppy, fr, grx, vnames, ifig)

    return ifig
end

function main(ifig)
    for sex in [2, 1]
        ifig = runx(sex, ifig)
    end
    return ifig
end

ifig = main(ifig)

f = [@sprintf("plots/%03d.pdf", j) for j = 0:ifig-1]
c = `gs -sDEVICE=pdfwrite -dNOPAUSE -dBATCH -dSAFER -sOutputFile=writing/nhanes_reg.pdf $f`
run(c)
