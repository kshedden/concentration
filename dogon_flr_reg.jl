#=
Functional low rank regression applied to the Dogon population, using
blood pressure as the dependent variable.
=#

using DataFrames, CSV, Printf, Statistics, UnicodePlots
using CodecZlib, PyPlot

rm("plots", recursive = true, force = true)
mkdir("plots")

ifig = 0

include("qreg_nn.jl")
include("flr_reg.jl")
include("basis.jl")
include("plot_utils.jl")

extra_vars =
    Dict(1 => Symbol[], 2 => Symbol[:lognummeas_BASE10, :F0_Mom_SBP_Z_Res_USE, :Bamako])

df = open("/home/kshedden/data/Beverly_Strassmann/Cohort_2021.csv.gz") do io
    CSV.read(GzipDecompressorStream(io), DataFrame)
end

# We want the first village where a person lived
df = sort(df, :datecomb)
vi = combine(groupby(df, :ID), :Village => first)
vi = rename(vi, :Village_first => :Village1)
df = leftjoin(df, vi, on = :ID)

# Drop people who are neither in the village nor in Bamako
df = filter(row -> !ismissing(row.Bamako) && row.Bamako in [0, 1], df)

# Indicator that a female is pregnant
df[!, :Preg] = df[:, :PregMo_Use] .> 0
df[!, :Preg] = replace(df[!, :Preg], true => 1, false => 0)

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

function make_dataframe(sex, cbs, mode, df)
    dx = filter(row -> row.Sex == sex, df)
    outcome = :SBP_MEAN
    vl =
        [:Age_Yrs, :BMI, :Ht_Ave_Use, Symbol(string(cbs) * "_pcscore"), extra_vars[mode]...]
    if sex == "Female"
        push!(vl, :Preg)
    end
    dr = dx[:, vcat([outcome], vl)]
    dr = dr[completecases(dr), :]
    for j = 1:size(dr, 2)
        dr[!, j] = Vector{Float64}(dr[:, j])
    end
    return dr, vl
end

# Standardize a variable so that its median value is zero
# and its scaled MAD is 1.  If Gaussian its standard deviation 
# is 1.
function zscore(x::AbstractVector)::AbstractVector
    x .-= median(x)
    x ./= median(abs.(x))
    x ./= 1.48
    return x
end

function do_all(dr, vl, sex, ifig)

    yv = Vector{Float64}(dr[:, :SBP_MEAN])
    xm = Matrix{Float64}(dr[:, vl])

    for (j, c) in enumerate(eachcol(xm))
        if length(unique(xm[:, j])) > 2
            xm[:, j] = zscore(c)
        end
    end

    # The quantile regression model.
    qr = qreg_nn(yv, xm)

    # Probability points
    pp = collect(range(0.1, 0.9, length = 9))

    # Fill in the quantiles and estimate the central axis
    qh = zeros(length(yv), length(pp))
    ca = zeros(length(pp))
    for (j, p) in enumerate(pp)
        _ = fit(qr, p, 0.1)
        qh[:, j] = qr.fit
        ca[j] = predict(qr, zeros(length(vl)))
    end

    # Remove the central axis
    qhc = copy(qh)
    for j in eachindex(ca)
        qhc[:, j] .-= ca[j]
    end

    # Build basis matrices for the low rank model
    X, Xp = Vector{Matrix{Float64}}(), Vector{Matrix{Float64}}()
    gr = collect(range(-2, 2, length = 101))
    grx = []
    for (j, x) in enumerate(eachcol(xm))
        if length(unique(x)) > 2
            # Not a binary variable
            xm[:, j] = zscore(x)
            B, g = genbasis(x, 5, std(x) / 2, linear = true)
            push!(X, B)
            push!(Xp, g(gr))
        else
            # A binary variable
            push!(X, x[:, :])
            bb = zeros(101)
            bb[51:end] .= 1
            push!(Xp, bb[:, :])
        end
        push!(grx, gr)
    end

    # Fit the low-rank model
    cu = 1000 * ones(length(X))
    cv = 1000 * ones(length(X))
    fr = fitlr(X, Xp, qhc, cu, cv)

    vnames = string.(vl)
    ifig = plots_flr(sex, X, Xp, pp, fr, grx, vnames, ifig)
    return ifig
end

function main(ifig)
    for cbs in [:HT, :WT, :BMI, :HAZ, :WAZ, :BAZ]
        for sex in ["Female", "Male"]
            for mode in [1, 2]
                println(cbs, " ", sex, " ", mode)
                dr, vl = make_dataframe(sex, cbs, mode, df)
                ifig = do_all(dr, vl, sex, ifig)
            end
        end
    end
    return ifig
end

ifig = main(ifig)

f = [@sprintf("plots/%03d.pdf", j) for j = 0:ifig-1]
c =
    `gs -sDEVICE=pdfwrite -dNOPAUSE -dBATCH -dSAFER -sOutputFile=writing/dogon_flr_reg.pdf $f`
run(c)
