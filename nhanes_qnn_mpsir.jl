using Statistics, DataFrames, Statistics, Random, Latexify
using PyPlot, Dimred, UnicodePlots, StatsBase, NearestNeighbors

rm("plots", recursive = true, force = true)
mkdir("plots")

ifig = 0

include("nhanes_prep.jl")
include("qreg_nn.jl")
include("cancorr.jl")
include("support.jl")

function run_mpsir(sex, npc, nmp)
    dx = select_sex(sex)
    y = dx[:, :BPXSY1]
    y = Vector{Float64}(y)
    xmat = dx[:, [:RIDAGEYR_z, :BMXBMI_z, :BMXHT_z]]
    xmat = Matrix{Float64}(xmat)
    return qnn_mpsir(y, xmat, npc, nmp)
end

# Probability points for SBP quantiles
pp = range(0.1, 0.9, length = 9)

# Use two MP-SIR factors.
nmp = 2

# Neighborhood size for local averaging
nnb = 50

function runx(sex, rslt, ifig)

    # Use one more PC than the number of MP-SIR factors.
    npc = nmp + 1

    eta, beta, qhc, xmat, eigx, eigy = run_mpsir(sex, npc, nmp)

    # A nearest-neighbor tree for finding neighbors in the 
    # projected x-space.
    xp = xmat * beta
    kt = KDTree(xp')

    # Plot the loading patterns for the SBP quantiles.
    PyPlot.clf()
    PyPlot.axes([0.16, 0.13, 0.72, 0.8])
    PyPlot.grid(true)
    PyPlot.title(sex == 1 ? "Male" : "Female")
    PyPlot.xlabel("Probability point", size = 15)
    PyPlot.ylabel("Loading", size = 15)
    for (j, e) in enumerate(eachcol(eta))
        PyPlot.plot(pp, e, "-", label = @sprintf("%d", j))
    end
    ha, lb = plt.gca().get_legend_handles_labels()
    leg = plt.figlegend(ha, lb, "center right")
    leg.draw_frame(false)
    PyPlot.savefig(@sprintf("plots/%03d.pdf", ifig))
    ifig += 1

    # Plot each Q score against each X score as a scatterplot
    for j1 = 1:nmp
        for j2 = 1:nmp
            PyPlot.clf()
            PyPlot.title(sex == 1 ? "Male" : "Female")
            PyPlot.grid(true)
            u1 = xmat * beta[:, j1]
            u2 = qhc * eta[:, j2]
            PyPlot.plot(u1, u2, "o", alpha = 0.2, rasterized = true)
            PyPlot.ylabel(@sprintf("Quantile score %d", j2), size = 15)
            PyPlot.xlabel(@sprintf("Covariate score %d", j1), size = 15)
            PyPlot.savefig(@sprintf("plots/%03d.pdf", ifig))
            ifig += 1
        end
    end

    # Get the support points, sort them by increasing x coordinate.
    u = xmat * beta
    sp = support([u[i, :] for i = 1:size(u, 1)], 5)
    z = [v[1] for v in sp]
    ii = sortperm(z)
    sp = [sp[i] for i in ii]

    # Plot the second X score against the first X score.  Show the support 
    # points with letters.
    PyPlot.clf()
    PyPlot.title(sex == 1 ? "Male" : "Female")
    PyPlot.grid(true)
    PyPlot.plot(u[:, 1], u[:, 2], "o", alpha = 0.2, rasterized = true)
    for (k, z) in enumerate(sp)
        PyPlot.text(
            z[1],
            z[2],
            string("ABCDEFGH"[k]),
            size = 14,
            ha = "center",
            va = "center",
        )
    end
    PyPlot.ylabel("Covariate score 2", size = 15)
    PyPlot.xlabel("Covariate score 1", size = 15)
    PyPlot.savefig(@sprintf("plots/%03d.pdf", ifig))
    ifig += 1

    # Plot the quantile trajectories corresponding to each letter
    # in the previous plot.
    PyPlot.clf()
    PyPlot.axes([0.12, 0.12, 0.75, 0.8])
    PyPlot.title(sex == 1 ? "Male" : "Female")
    for (j, z) in enumerate(sp)

        # Nearest neighbors of the support point in the projected
        # X-space.
        ii, _ = knn(kt, z, nnb)

        row = [
            sex == 1 ? "Male" : "Female",
            string("ABCDEFGH"[j]),
            mean(xmat[ii, :], dims = 1)...,
        ]
        push!(rslt, row)

        qp = mean(qhc[ii, :], dims = 1)
        PyPlot.plot(pp, vec(qp), "-", label = string("ABCDEFGH"[j]))
    end
    ha, lb = PyPlot.gca().get_legend_handles_labels()
    leg = PyPlot.figlegend(ha, lb, "center right")
    leg.draw_frame(false)
    PyPlot.grid(true)
    PyPlot.xlabel("Probability", size = 15)
    PyPlot.ylabel("SBP quantile deviation", size = 15)
    PyPlot.savefig(@sprintf("plots/%03d.pdf", ifig))
    ifig += 1

    return ifig, rslt
end

function main(ifig)

    rslt = DataFrame(
        Sex = String[],
        Point = String[],
        Age = Float64[],
        BMI = Float64[],
        Height = Float64[],
    )

    for sex in [2, 1]
        ifig, rslt = runx(sex, rslt, ifig)
    end
    return rslt, ifig
end

rslt, ifig = main(ifig)

open("writing/nhanes_qnn_mpsir_table1.tex", "w") do io
    write(io, latexify(rslt, fmt = "%.2f", env = :table))
end

f = [@sprintf("plots/%03d.pdf", j) for j = 0:ifig-1]
c =
    `gs -sDEVICE=pdfwrite -dNOPAUSE -dBATCH -dSAFER -sOutputFile=writing/nhanes_qnn_mpsir_loadings.pdf $f`
run(c)
