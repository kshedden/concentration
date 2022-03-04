using Statistics, DataFrames, Statistics, Random, Latexify
using PyPlot, NearestNeighbors

rm("plots", recursive = true, force = true)
mkdir("plots")

ifig = 0

include("utils.jl")
include("nhanes_prep.jl")
include("cancorr.jl")
include("plot_utils.jl")

function run_cca(sex, npc, nperm)
    dx = select_nhanes(sex, 18, 40)
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
nnb = 100

function runx(sex, npc, rslt, rsltp, ifig)

    sexs = sex == 2 ? "Female" : "Male"
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

    for vx = 1:3

        # Get the support points, sort them by increasing x coordinate.
        xp = xmat * beta
        _, spt, sptl = make_support(xp, beta, vx, 4)

        # A nearest-neighbor tree for finding neighborhoods in the
        # projected X-space.
        kt = KDTree(xp')

        # Plot the X scores against each other.  Show the support 
        # points with letters.
        vname = ["Age", "BMI", "Height"][vx]
        ifig = plotxx_all(sexs, vname, xp, spt, sptl, beta, ifig)

        # Plot the quantile trajectories corresponding to each letter
        # in the previous plot.
        rsltp1, ifig = plot_qtraj(sexs, npc, vname, spt, sptl, kt, xmat, qhc, nnb, ifig)
        ifig = plot_qtraj_diff(sexs, npc, vname, spt, sptl, kt, xmat, qhc, nnb, ifig)

        if vx == 1
            for row in rsltp1
                push!(rsltp, row)
            end
        end
    end

    # Save the coefficient estimates and correlation values
    for (j, c) in enumerate(eachcol(beta))
        row = [sexs, @sprintf("%d", npc), @sprintf("%d", j)]
        for a in c
            push!(row, @sprintf("%.2f", a))
        end
        push!(row, @sprintf("%.2f", ss[j]))
        push!(row, @sprintf("%.2f", quantile(sp[j, :], 0.95)))
        push!(rslt, row)
    end

    return rslt, rsltp, beta, eta, ifig
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
        sexs = sex == 2 ? "Female" : "Male"
        println("sex=$(sex) $(sexs)")
        for npc in [1, 2, 3]
            rslt, rsltp, beta, eta, ifig = runx(sex, npc, rslt, rsltp, ifig)

            # Save these for comparison between CCA and MP-SIR
            if npc == 2
                for (j, c) in enumerate(eachcol(beta))
                    row = [sexs, j, beta[:, j]...]
                    push!(beta_x2, row)
                end
                for (j, c) in enumerate(eachcol(eta))
                    row = [sexs, j, eta[:, j]...]
                    push!(eta_x2, row)
                end
            end
        end
    end

    CSV.write("beta_cca.csv", beta_x2)
    CSV.write("eta_cca.csv", eta_x2)

    return ifig, rslt, rsltp
end

ifig, rslt, rsltp = main(ifig)

# DEBUG leading underscore in file name
open("writing/_nhanes_qnn_cca_table1.tex", "w") do io
    write(io, latexify(rslt, env = :table))
end

open("writing/nhanes_qnn_cca_table2.tex", "w") do io
    write(io, latexify(rsltp, env = :table, fmt = "%.2f"))
end

f = [@sprintf("plots/%03d.pdf", j) for j = 0:ifig-1]
c =
    `gs -sDEVICE=pdfwrite -dNOPAUSE -dBATCH -dSAFER -sOutputFile=writing/nhanes_qnn_cca_loadings.pdf $f`
run(c)
