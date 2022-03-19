#=
Use GEE to model Dogon SBP in terms of current height and BMI, 
early childhood growth (PC-scores), and control variables.
=#

using CSV, DataFrames, CodecZlib, Printf, Statistics, UnicodePlots
using StatsBase, StatsModels, Distributions, GLM, GEE, LinearAlgebra
using PyPlot

df = open("/home/kshedden/data/Beverly_Strassmann/Cohort_2021.csv.gz") do io
    CSV.read(GzipDecompressorStream(io), DataFrame)
end

# Indicator that a female is pregnant
df[!, :Preg] = df[:, :PregMo_Use] .> 0
df[!, :Preg] = replace(df[!, :Preg], true => 1, false => 0)

rslt = DataFrame(
    :Sex => String[],
    :CBSVar => String[],
    :Adj => Int[],
    :BMI => Float64[],
    :BMI_se => Float64[],
    :Ht => Float64[],
    :Ht_se => Float64[],
    :CBS => Float64[],
    :CBS_se => Float64[],
)

function merge_pc_scores(df)
    pcs = open("dogon_pc_scores.csv") do io
        CSV.read(io, DataFrame)
    end
    pcs = rename(pcs, :id => :ID)
    na = names(pcs)[2:end]
    for a in na
        pcs = rename(pcs, a => Symbol(@sprintf("%s_pcscore", string(a))))
    end
    df = outerjoin(df, pcs, on = :ID)
    return df
end

df = merge_pc_scores(df)

# We want the first village where a person lived
df = sort(df, :datecomb)
vi = combine(groupby(df, :ID), :Village => first)
vi = rename(vi, :Village_first => :Village1)
df = leftjoin(df, vi, on = :ID)

# Drop people who are neither in the village nor in Bamako
df = filter(row -> !ismissing(row.Bamako) && row.Bamako in [0, 1], df)

# Logarithm of the number of prior BP measurements
df[:, :lognummeas] = df[:, :lognummeas_BASE10]

function zscore(x::AbstractVector)::AbstractVector
    x .-= median(x)
    x ./= median(abs.(x))
    x ./= 1.48
    return x
end

function check_regression(sex, cbs, adj, df, out)
    outcome = "SBP_MEAN"
    cbsv = string(cbs) * "_pcscore"
    xvars = ["BMI", "Ht_Ave_Use", cbsv, "Age_Yrs"]
    if sex == "Female"
        push!(xvars, "Preg")
    end
    if adj == 1
        push!(xvars, "Village1", "lognummeas")
    end

    dr = filter(row -> row.Sex == sex, df)
    dr = dr[:, vcat([outcome, "ID"], xvars)]
    dr = dr[completecases(dr), :]
    for j = 1:size(dr, 2)
        dr[!, j] = Vector{Float64}(dr[:, j])
    end

    # Z-score non-categorical variables
    for x in xvars
        if !(x in ["Village1", "Bamako", "Preg"])
            dr[:, x] = zscore(dr[:, x])
        end
    end

    # Polynomial terms in age
    for k = 2:3
        s = "Age_Yrs$(k)"
        dr[:, s] = dr[:, :Age_Yrs] .^ k
        push!(xvars, s)
    end

    fml = Term(:SBP_MEAN) ~ sum(Term.(Symbol.(xvars)))
    dr = sort(dr, :ID)
    contrasts = Dict(:Village1 => DummyCoding())
    mm = gee(
        fml,
        dr,
        dr[:, :ID],
        Normal(),
        ExchangeableCor(),
        IdentityLink(),
        contrasts = contrasts,
    )

    s = split(string(mm), "\n")
    s = join(s[6:end], "\n")
    write(out, @sprintf("%s %s\n", sex, string(cbs)))
    write(out, @sprintf("n=%d\n", nobs(mm)))
    write(out, s)
    write(out, "\n\n")

    c = coef(mm)
    s = sqrt.(diag(vcov(mm)))
    row = [sex, string(cbs), adj, c[2], s[2], c[3], s[3], c[4], s[4]]
    push!(rslt, row)
end

function make_dotplot()

    y = 0
    e = 0.1
    xlim = [-4, 10]
    ms = 4
    mfc = "white"

    PyPlot.figure(figsize = (8, 12), frameon = false)
    PyPlot.clf()
    ax = PyPlot.axes([0.1, 0.1, 0.8, 0.83])
    ticker = PyPlot.matplotlib.ticker
    ax.yaxis.set_major_locator(ticker.NullLocator())
    ax.spines["left"].set_visible(false)
    ax.spines["right"].set_visible(false)
    ax.spines["top"].set_visible(false)

    for jj = 1:12

        if (jj - 1) % 4 == 0
            r = PyPlot.matplotlib.patches.Rectangle(
                [xlim[1], y - 0.5],
                xlim[2] - xlim[1],
                2,
                alpha = 0.2,
                facecolor = "lightgrey",
                edgecolor = "lightgrey",
            )
            ax.add_patch(r)
        end

        PyPlot.plot(xlim, [y, y], "-", color = "grey", lw = 2)

        row = rslt[2*(jj-1)+1, :]
        PyPlot.text(xlim[1] - 0.5, y, string(row[:CBSVar]), ha = "center", va = "center")
        if (jj - 1) % 2 == 1
            PyPlot.text(xlim[2] + 0.5, y, "A", ha = "center", va = "center")
        end

        # Female
        row = rslt[2*(jj-1)+1, :]
        PyPlot.plot(
            [row.Ht - 2 * row.Ht_se, row.Ht + 2 * row.Ht_se],
            [y + e, y + e],
            "-",
            color = "grey",
        )
        kwds = jj == 1 ? Dict(:label => "Female adult height") : Dict()
        PyPlot.plot(
            [row.Ht, row.Ht],
            [y + e, y + e],
            "v",
            color = "orange",
            ms = ms,
            mfc = mfc;
            kwds...,
        )
        PyPlot.plot(
            [row.BMI - 2 * row.BMI_se, row.BMI + 2 * row.BMI_se],
            [y + 2 * e, y + 2 * e],
            "-",
            color = "grey",
        )
        kwds = jj == 1 ? Dict(:label => "Female adult BMI") : Dict()
        PyPlot.plot(
            [row.BMI, row.BMI],
            [y + 2 * e, y + 2 * e],
            "o",
            color = "orange",
            ms = ms,
            mfc = mfc;
            kwds...,
        )
        PyPlot.plot(
            [row.CBS - 2 * row.CBS_se, row.CBS + 2 * row.CBS_se],
            [y + 3 * e, y + 3 * e],
            "-",
            color = "grey",
        )
        kwds = jj == 1 ? Dict(:label => "Female childhood body size") : Dict()
        PyPlot.plot(
            [row.CBS, row.CBS],
            [y + 3 * e, y + 3 * e],
            "^",
            color = "orange",
            ms = ms,
            mfc = mfc;
            kwds...,
        )

        # Male
        row = rslt[2*(jj-1)+2, :]
        PyPlot.plot(
            [row.Ht - 2 * row.Ht_se, row.Ht + 2 * row.Ht_se],
            [y - e, y - e],
            "-",
            color = "grey",
        )
        kwds = jj == 1 ? Dict(:label => "Male adult height") : Dict()
        PyPlot.plot(
            [row.Ht, row.Ht],
            [y - e, y - e],
            "v",
            color = "purple",
            ms = ms,
            mfc = mfc;
            kwds...,
        )
        PyPlot.plot(
            [row.BMI - 2 * row.BMI_se, row.BMI + 2 * row.BMI_se],
            [y - 2 * e, y - 2 * e],
            "-",
            color = "grey",
        )
        kwds = jj == 1 ? Dict(:label => "Male adult BMI") : Dict()
        PyPlot.plot(
            [row.BMI, row.BMI],
            [y - 2 * e, y - 2 * e],
            "o",
            color = "purple",
            ms = ms,
            mfc = mfc;
            kwds...,
        )
        PyPlot.plot(
            [row.CBS - 2 * row.CBS_se, row.CBS + 2 * row.CBS_se],
            [y - 3 * e, y - 3 * e],
            "-",
            color = "grey",
        )
        kwds = jj == 1 ? Dict(:label => "Male childhood body size") : Dict()
        PyPlot.plot(
            [row.CBS, row.CBS],
            [y - 3 * e, y - 3 * e],
            "^",
            color = "purple",
            ms = ms,
            mfc = mfc;
            kwds...,
        )

        y += 1
    end

    PyPlot.axvline(0, color = "black")

    ha, lb = ax.get_legend_handles_labels()
    leg = PyPlot.figlegend(ha, lb, "upper center", ncol = 2)
    leg.draw_frame(false)

    PyPlot.xlabel("Standardized coefficient", size = 15)
    PyPlot.savefig("writing/gee_dotplot.pdf")
end

out = open("writing/gee_regression.txt", "w")

for cbs in [:HT, :WT, :BMI, :HAZ, :WAZ, :BAZ]
    for adj in [0, 1]
        check_regression("Female", cbs, adj, df, out)
        check_regression("Male", cbs, adj, df, out)
    end
end

close(out)

make_dotplot()
