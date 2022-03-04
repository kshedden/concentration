#=
Functional low rank regression applied to the Dogon population, using
blood pressure as the dependent variable.
=#

using DataFrames, CSV, Printf, Statistics, UnicodePlots
using CodecZlib, PyPlot

rm("plots", recursive = true, force = true)
mkdir("plots")

ifig = 0

include("flr_reg.jl")
include("basis.jl")
include("plot_utils.jl")
include("dogon_prep.jl")

extra_vars =
    Dict(1 => Symbol[], 2 => Symbol[:lognummeas_BASE10, :F0_Mom_SBP_Z_Res_USE, :Bamako])

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
    qr = qregnn(yv, xm)

    # Probability points
    pp = collect(range(0.1, 0.9, length = 9))

    # Fill in the quantiles and estimate the central axis
    qh = zeros(length(yv), length(pp))
    ca = zeros(length(pp))
    for (j, p) in enumerate(pp)
        fit!(qr, p)
        qh[:, j] = fittedvalues(qr)
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
                dr, vl = make_dogon_dataframe(sex, cbs, extra_vars[mode], df)
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
