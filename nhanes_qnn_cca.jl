using Statistics, DataFrames, Statistics, Random, Latexify
using PyPlot, NearestNeighbors

rm("plots", recursive = true, force = true)
mkdir("plots")

ifig = 0

include("nhanes_prep.jl")
include("qreg_nn.jl")
include("cancorr.jl")
include("support.jl")

function run_cca(sex, npc, nperm)
    dx = select_sex(sex)
    y = dx[:, :BPXSY1]
    y = Vector{Float64}(y)
    xmat = dx[:, [:RIDAGEYR_z, :BMXBMI_z, :BMXHT_z]]
    xmat = Matrix{Float64}(xmat)
    return qnn_cca(y, xmat, npc, nperm)
end

# Probability points for SBP quantiles
pp = range(0.1, 0.9, length = 9)

# Number of permutations for stability analysis
nperm = 100

# Number of neighbors for local averaging
nnb = 50

function runx(sex, npc, rslt, rsltp, ifig)

    eta, beta, qhc, xmat, ss, sp = run_cca(sex, npc, nperm)

    # Plot the quantile loading patterns
    PyPlot.clf()
    PyPlot.axes([0.13, 0.1, 0.75, 0.8])
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
    for j1 = 1:size(beta, 2)
        for j2 = 1:size(eta, 2)
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
    xp = xmat * beta
    spt = support([xp[i, :] for i = 1:size(xp, 1)], 5)
    z = [v[1] for v in spt]
    ii = sortperm(z)
    spt = [spt[i] for i in ii]

	# A nearest-neighbor tree for finding neighborhoods in the
	# projected X-space.
    kt = KDTree(xp')

    # Plot the X scores against each other.  Show the support 
    # points with letters.
    for j2 = 1:size(beta, 2)
        for j1 = 1:j2-1
            PyPlot.clf()
            PyPlot.title(sex == 1 ? "Male" : "Female")
            PyPlot.grid(true)
            PyPlot.plot(xp[:, j1], xp[:, j2], "o", alpha = 0.2, rasterized = true)
            for (k, z) in enumerate(spt)
                PyPlot.text(
                    z[j1],
                    z[j2],
                    string("ABCDEFGH"[k]),
                    size = 14,
                    ha = "center",
                    va = "center",
                )
            end
            PyPlot.ylabel(@sprintf("Covariate score %d", j2), size = 15)
            PyPlot.xlabel(@sprintf("Covariate score %d", j1), size = 15)
            PyPlot.savefig(@sprintf("plots/%03d.pdf", ifig))
            ifig += 1
        end
    end

    # Plot the quantile trajectories corresponding to each letter
    # in the previous plot.
    PyPlot.clf()
    PyPlot.axes([0.12, 0.12, 0.75, 0.8])
    PyPlot.title(sex == 1 ? "Male" : "Female")
    for (j, z) in enumerate(spt)

        # Nearest neighbors of the support point in the projected
        # X-space.
        ii, _ = knn(kt, z, nnb)

		# Store the x-variable means corresponding to
		# each support point.
        row = [
            sex == 1 ? "Male" : "Female",
            npc,
            string("ABCDEFGH"[j]),
            mean(xmat[ii, :], dims = 1)...,
        ]
        push!(rsltp, row)

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

	# Save the estimates and correlation values
    for (j, c) in enumerate(eachcol(beta))
        row = [sex == 2 ? "Female" : "Male", @sprintf("%d", npc), @sprintf("%d", j)]
        for a in c
            push!(row, @sprintf("%.2f", a))
        end
        push!(row, @sprintf("%.2f", ss[j]))
        push!(row, @sprintf("%.2f", quantile(sp[j, :], 0.95)))
        push!(rslt, row)
    end

    return rslt, rsltp, ifig
end

function main(ifig)

    rslt = DataFrame(
        sex = String[],
        npc = String[],
        c = String[],
        Age = String[],
        BMI = String[],
        Height = String[],
        R = String[],
        Rp = String[],
    )

	rsltp = DataFrame(
		Sex = String[],
		NPC = Int[],
		Point = String[],
		Age = Float64[],
		BMI = Float64[],
		Height = Float64[])

    for sex in [2, 1]
        println("sex=$(sex)")
        for npc in [1, 2, 3]
            rslt, rsltp, ifig = runx(sex, npc, rslt, rsltp, ifig)
        end
    end

    return ifig, rslt, rsltp
end

ifig, rslt, rsltp = main(ifig)

open("writing/nhanes_qnn_cca_table1.tex", "w") do io
    write(io, latexify(rslt, env = :table))
end

open("writing/nhanes_qnn_cca_table2.tex", "w") do io
    write(io, latexify(rsltp, env = :table))
end

f = [@sprintf("plots/%03d.pdf", j) for j = 0:ifig-1]
c =
    `gs -sDEVICE=pdfwrite -dNOPAUSE -dBATCH -dSAFER -sOutputFile=writing/nhanes_qnn_cca_loadings.pdf $f`
run(c)
