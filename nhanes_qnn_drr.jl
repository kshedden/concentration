using Statistics, DataFrames, Statistics, Random, Latexify
using PyPlot

rm("plots", recursive = true, force = true)
mkdir("plots")

ifig = 0

include("nhanes_prep.jl")
include("qreg_nn.jl")
include("cancorr.jl")

# Attach a Z-scored version of the variable named 'vn' to
# the dataframe df.
function zscore!(df, vn)
    vz = Symbol(string(vn) * "_z")
    df[!, vz] = (df[:, vn] .- mean(df[:, vn])) ./ std(df[:, vn])
end

function select_sex(sex)
    dx = df[df.RIAGENDR.==sex, :]
    dx = dx[:, [:BPXSY1, :BMXBMI, :RIDAGEYR, :BMXHT]]
    dx = dx[completecases(dx), :]
    zscore!(dx, :RIDAGEYR)
    zscore!(dx, :BMXBMI)
    zscore!(dx, :BMXHT)
    return dx
end

function run1(sex, npc, nperm)
    dx = select_sex(sex)
    y = dx[:, :BPXSY1]
    y = Vector{Float64}(y)
    xmat = dx[:, [:RIDAGEYR_z, :BMXBMI_z, :BMXHT_z]]
    xmat = Matrix{Float64}(xmat)
    return qnn_cca(y, xmat, npc, nperm)
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
        for npc in [1, 2, 3]
            eta, beta, qhc, xmat, ss, sp = run1(sex, npc, nrand)

            PyPlot.clf()
            PyPlot.axes([0.13, 0.1, 0.75, 0.8])
            PyPlot.grid(true)
            PyPlot.title(sex == 1 ? "Male" : "Female")
            PyPlot.xlabel("Probability point", size=15)
            PyPlot.ylabel("Loading", size=15)
            for (j, e) in enumerate(eachcol(eta))
                PyPlot.plot(pp, e, "-", label = @sprintf("%d", j))
            end
            ha, lb = plt.gca().get_legend_handles_labels()
            leg = plt.figlegend(ha, lb, "center right")
            leg.draw_frame(false)
            PyPlot.savefig(@sprintf("plots/%03d.pdf", ifig))
            ifig += 1

            println(ss)
            println(maximum(sp))
            for (j, c) in enumerate(eachcol(beta))
                row = [sex == 2 ? "Female" : "Male", @sprintf("%d", npc), @sprintf("%d", j)]
                for a in c
                    push!(row, @sprintf("%.2f", a))
                end
                push!(row, @sprintf("%.2f", ss[j]))
                push!(row, @sprintf("%.2f", quantile(sp[j, :], 0.95)))
                push!(rslt, row)
            end
        end
    end
    
    return ifig, rslt
end

ifig, rslt = main(ifig)

open("writing/nhanes_qnn_drr_table1.tex", "w") do io
    write(io, latexify(rslt, env = :table))
end

f = [@sprintf("plots/%03d.pdf", j) for j = 0:ifig-1]
c =
    `gs -sDEVICE=pdfwrite -dNOPAUSE -dBATCH -dSAFER -sOutputFile=writing/nhanes_qnn_drr_loadings.pdf $f`
run(c)
