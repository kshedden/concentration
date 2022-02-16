using Statistics, DataFrames, Statistics, Random, Latexify
using PyPlot, Dimred, UnicodePlots, StatsBase

rm("plots", recursive = true, force = true)
mkdir("plots")

ifig = 0

include("nhanes_prep.jl")
include("qreg_nn.jl")
include("cancorr.jl")

function run_mpsir(sex, npc, nms, nperm)
    dx = select_sex(sex)
    y = dx[:, :BPXSY1]
    y = Vector{Float64}(y)
    xmat = dx[:, [:RIDAGEYR_z, :BMXBMI_z, :BMXHT_z]]
    xmat = Matrix{Float64}(xmat)
    return qnn_mpsir(y, xmat, npc, nms, nperm)
end

pp = range(0.1, 0.9, length = 9)

function main(ifig)

    nrand = 1

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
    for sex in [2, 1]
        println("sex=$(sex)")
        for nms in [2]
            npc = nms + 1
            eta, beta, qhc, xmat, eigx, eigy = run_mpsir(sex, npc, nms, nrand)

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

            # Q vs X scatterplots
            for j1 = 1:nms
                for j2 = 1:nms
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

            # X versus X scatterplot
            PyPlot.clf()
            PyPlot.title(sex == 1 ? "Male" : "Female")
            PyPlot.grid(true)
            u1 = xmat * beta[:, 1]
            u2 = xmat * beta[:, 2]
            println(mean(u1), " ", mean(u2), " ", std(u1), " ", std(u2))
            PyPlot.plot(u1, u2, "o", alpha = 0.2, rasterized = true)
            PyPlot.ylabel("Covariate score 2", size = 15)
            PyPlot.xlabel("Covariate score 1", size = 15)
            PyPlot.savefig(@sprintf("plots/%03d.pdf", ifig))
            ifig += 1
        end
    end

    return ifig, rslt
end

ifig, rslt = main(ifig)

open("writing/nhanes_qnn_mpsir_table1.tex", "w") do io
    write(io, latexify(rslt, env = :table))
end

f = [@sprintf("plots/%03d.pdf", j) for j = 0:ifig-1]
c =
    `gs -sDEVICE=pdfwrite -dNOPAUSE -dBATCH -dSAFER -sOutputFile=writing/nhanes_qnn_mpsir_loadings.pdf $f`
run(c)
