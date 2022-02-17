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

# Neighborhood size for local averaging
nnb = 50

function runx(sex, nmp, rslt, ifig)

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
    xp = xmat * beta
    sp = support([xp[i, :] for i = 1:size(xp, 1)], 6)
    z = [v[1] for v in sp]
    ii = sortperm(z)
    sp = [sp[i] for i in ii]

    # Make a score plot for each pair of X-side factors.  Show the support 
    # points with letters.
    for j2 = 1:size(beta, 2)
        for j1 = 1:j2-1
            PyPlot.clf()
            PyPlot.title(sex == 1 ? "Male" : "Female")
            PyPlot.grid(true)
            PyPlot.plot(xp[:, j1], xp[:, j2], "o", alpha = 0.2, rasterized = true)

            # Make it a biplot
            bs = 2 * beta[:, [j1, j2]]
            for j = 1:3
                # Move the text so that the arrow ends at the loadings.
                bs[j, :] .+= 0.2 * bs[j, :] / norm(bs[j, :])
            end
            PyPlot.gca().annotate(
                "Age",
                xytext = (bs[1, 1], bs[1, 2]),
                xy = (0, 0),
                arrowprops = Dict(:arrowstyle => "<-", :shrinkA => 0, :shrinkB => 0),
            )
            PyPlot.gca().annotate(
                "BMI",
                xytext = (bs[2, 1], bs[2, 2]),
                xy = (0, 0),
                arrowprops = Dict(:arrowstyle => "<-", :shrinkA => 0, :shrinkB => 0),
            )
            PyPlot.gca().annotate(
                "Ht",
                xytext = (bs[3, 1], bs[3, 2]),
                xy = (0, 0),
                arrowprops = Dict(:arrowstyle => "<-", :shrinkA => 0, :shrinkB => 0),
            )

            # Show the support points with letters.
            for (k, z) in enumerate(sp)
                PyPlot.text(
                    z[j1],
                    z[j2],
                    string("ABCDEFGH"[k]),
                    size = 14,
                    ha = "center",
                    va = "center",
                )
            end
            PyPlot.ylabel("Covariate score $(j2)", size = 15)
            PyPlot.xlabel("Covariate score $(j1)", size = 15)
            PyPlot.savefig(@sprintf("plots/%03d.pdf", ifig))
            ifig += 1
        end
    end

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

    return rslt, beta, eta, ifig
end

function main(ifig)

    rslt = DataFrame(
        Sex = String[],
        Point = String[],
        Age = Float64[],
        BMI = Float64[],
        Height = Float64[],
    )

    beta_x2 = DataFrame(
        Sex = String[],
        J = Int[],
        Age = Float64[],
        BMI = Float64[],
        Height = Float64[],
    )
    eta_x2 = DataFrame(
        Sex = String[],
        J = Int[],
        eta1 = Float64[],
        eta2 = Float64[],
        eta3 = Float64[],
        eta4 = Float64[],
        eta5 = Float64[],
        eta6 = Float64[],
        eta7 = Float64[],
        eta8 = Float64[],
        eta9 = Float64[],
    )

    for sex in [2, 1]
        for nmp in [2, 3]
            rslt, beta, eta, ifig = runx(sex, nmp, rslt, ifig)

            # Save coefficients for comparison to CCA
            if nmp == 2
                for (j, c) in enumerate(eachcol(beta))
                    row = [sex == 2 ? "Female" : "Male", j, beta[:, j]...]
                    push!(beta_x2, row)
                end
                for (j, c) in enumerate(eachcol(eta))
                    row = [sex == 2 ? "Female" : "Male", j, eta[:, j]...]
                    push!(eta_x2, row)
                end
            end
        end
    end

    CSV.write("beta_mpsir.csv", beta_x2)
    CSV.write("eta_mpsir.csv", eta_x2)

    return rslt, ifig
end

rslt, ifig = main(ifig)

open("writing/nhanes_qnn_mpsir_table1.tex", "w") do io
    write(io, latexify(rslt, fmt = "%.2f", env = :table))
end

CSV.write("nhanes_qnn_mpsir.csv", rslt)

f = [@sprintf("plots/%03d.pdf", j) for j = 0:ifig-1]
c =
    `gs -sDEVICE=pdfwrite -dNOPAUSE -dBATCH -dSAFER -sOutputFile=writing/nhanes_qnn_mpsir_loadings.pdf $f`
run(c)
