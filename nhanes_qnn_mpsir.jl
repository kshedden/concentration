using Statistics, DataFrames, Statistics, Random, Latexify
using PyPlot, Dimred, UnicodePlots, StatsBase, NearestNeighbors

rm("plots", recursive = true, force = true)
mkdir("plots")

ifig = 0

include("nhanes_prep.jl")
include("qreg_nn.jl")
include("cancorr.jl")
include("support.jl")
include("plot_utils.jl")

function run_mpsir(sex, npc, nmp)
    dx = select_nhanes(sex, 18, 40)
    y = dx[:, :BPXSY1]
    y = Vector{Float64}(y)
    xmat = dx[:, [:RIDAGEYR_z, :BMXBMI_z, :BMXHT_z]]
    xmat = Matrix{Float64}(xmat)
    return qnn_mpsir(y, xmat, npc, nmp)
end

# Probability points for SBP quantiles
pp = range(0.1, 0.9, length = 9)

# Neighborhood size for local averaging
nnb = 100

function runx(sex, nmp, rslt, ifig)

    # Use one more PC than the number of MP-SIR factors.
    npc = nmp + 1

    eta, beta, qhc, xmat, eigx, eigy = run_mpsir(sex, npc, nmp)

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

    for vx = 1:3

        # A nearest-neighbor tree for finding neighbors in the 
        # projected x-space.
        xp = xmat * beta
        kt = KDTree(xp')

        spt, sptl = make_support(xp, beta, vx, 4)

        # Plot the X scores against each other.  Show the support 
        # points with letters.
        vname = ["Age", "BMI", "Height"][vx]
        ifig = plotxx_all(sex, vname, xp, spt, sptl, beta, ifig)

        # Plot the quantile trajectories corresponding to each letter
        # in the previous plot.
        _, ifig = plot_qtraj(sex, npc, vname, spt, sptl, kt, xmat, qhc, ifig)
        ifig = plot_qtraj_diff(sex, npc, vname, spt, sptl, kt, xmat, qhc, ifig)
    end

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
