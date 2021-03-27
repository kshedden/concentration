# Get these NHANES data files:
#
# wget https://wwwn.cdc.gov/Nchs/Nhanes/2015-2016/DEMO_I.XPT
# wget https://wwwn.cdc.gov/Nchs/Nhanes/2015-2016/BPX_I.XPT
# wget https://wwwn.cdc.gov/Nchs/Nhanes/2015-2016/BMX_I.XPT
#
# Use Python to convert from SAS:
#
# import pandas as pd
# for x in ("DEMO", "BPX", "BMX"): pd.read_sas("%s_I.XPT" % x).to_csv("%s_I.csv.gz" % x, index=None)

using DataFrames, CSV, GZip, Printf, UnicodePlots, LinearAlgebra

include("qreg_nn.jl")
include("functional_lr.jl")

dx = []
for x in ["DEMO", "BPX", "BMX"]
    d = GZip.open(@sprintf("%s_I.csv.gz", x)) do io
        CSV.read(io, DataFrame)
    end
    push!(dx, d)
end

df = outerjoin(dx[1], dx[2], on=:SEQN)
df = outerjoin(df, dx[3], on=:SEQN)

da = df[:, [:RIAGENDR, :RIDAGEYR, :BPXSY1, :BMXBMI]]
da = da[completecases(da), :]

# Standardize column v of da, and return functions that
# can be used to standardize and de-standardize other
# data relative to this scale.
function standardize!(da, v)
    u = Symbol(@sprintf("%s_z", string(v)))
    m = mean(da[:, v])
    s = std(da[:, v])
    da[!, u] = (da[:, v] .- m) ./ s
    return tuple(x->(x-m)/s, x->x*s+m)
end

male = da[da.RIAGENDR .== 1, :]
female = da[da.RIAGENDR .== 2, :]

age_f_x, age_f_r = standardize!(female, :RIDAGEYR)
age_m_x, age_m_r = standardize!(male, :RIDAGEYR)
bmi_f_x, bmi_f_r = standardize!(female, :BMXBMI)
bmi_m_x, bmi_m_r = standardize!(male, :BMXBMI)


# Looks like lambda=0.2 works OK
function check_tuning1(dx, p)

    y = Array{Float64,1}(dx[:, :BMXBMI_z])
    x = Array{Float64,1}(dx[:, :RIDAGEYR_z])[:, :]
    nn = qreg_nn(y, x)

    lam = range(0.01, 0.5, length=10)
    b = zeros(length(lam))
    for (i, la) in enumerate(lam)
        _ = fit(nn, p, la)
        b[i] = bic(nn)
    end

    plt = lineplot(lam, b)
    println(plt)

end

# Looks like lambda=0.2 works OK
function check_tuning2(dx, p)

    y = Array{Float64,1}(dx[:, :BPXSY1])
    x = Array{Float64,2}(dx[:, [:RIDAGEYR_z, :BMXBMI_z]])
    nn = qreg_nn(y, x)

    lam = range(0.01, 1, length=10)
    b = zeros(length(lam))
    for (i, la) in enumerate(lam)
        _ = fit(nn, p, la)
        b[i] = bic(nn)
    end

    plt = lineplot(lam, b)
    println(plt)

end

for sex in ["female", "male"]

    if sex == "female"
        dx = female
        age_x = age_f_x
        age_r = age_f_r
        bmi_x = bmi_f_x
    else
        dx = male
        age_x = age_m_x
        age_r = age_m_r
        bmi_x = bmi_m_x
    end

    y = Array{Float64,1}(dx[:, :BMXBMI])
    x = Array{Float64,1}(dx[:, :RIDAGEYR_z])[:, :]
    nn0 = qreg_nn(y, x)

    y = Array{Float64,1}(dx[:, :BPXSY1])
    x = Array{Float64,2}(dx[:, [:RIDAGEYR_z, :BMXBMI_z]])
    nn = qreg_nn(y, x)

    ages = [25., 50., 75.]
    plt0, plt1, plt2 = nothing, nothing, nothing

    for age in ages

        # Get the BMI quantiles
        pr = range(0.1, 0.9, length=20)
        bmi = zeros(length(pr))
        for (i, p) in enumerate(pr)
            fit(nn0, p, 0.2)
            bmi[i] = predict_smooth(nn0, [age_x(age)], [0.5])
        end

        # Plot the BMI quantiles
        if age == ages[1]
            plt0 = lineplot(pr, bmi, xlabel="Probability", ylabel="BMI",
                           title=sex, name=@sprintf("Age=%.0f", age))
        else
            lineplot!(plt0, pr, bmi, name=@sprintf("Age=%.0f", age))
        end

        # Estimate SBP quantiles at each BMI quantile
        sbp = zeros(length(pr), length(pr))
        for (i2, p2) in enumerate(pr)
            fit(nn, p2, 0.2)
            for (i1, p1) in enumerate(pr)
                sbp[i1, i2] = predict_smooth(nn, [age_x(age), bmi_x(bmi[i1])], [0.5, 0.5])
            end
        end

        # The median point of the covariate (BMI)
        ii = div(size(sbp, 1), 2)

        # Plot SBP quantiles at median BMI, for specific age/sex groups
        if age == ages[1]
            plt1 = lineplot(pr, sbp[ii, :], xlabel="SBP probability point", ylabel="SBP at median BMI",
                           title=sex, ylim=[100, 150], name=@sprintf("Age=%.0f", age))
        else
            lineplot!(plt1, pr, sbp[ii, :], name=@sprintf("Age=%.0f", age))
        end

        # Plot median SBP at each BMI quantile, for specific age/sex groups
        if age == ages[1]
            plt2 = lineplot(pr, sbp[:, ii], xlabel="BMI probability point", ylabel="SBP",
                           title=sex, ylim=[100, 150], name=@sprintf("Age=%.0f", age))
        else
            lineplot!(plt2, pr, sbp[:, ii], name=@sprintf("Age=%.0f", age))
        end

        # Center the SBP quantiles around the median BMI
        sbpc = copy(sbp)
        for j in 1:size(sbp, 2)
            sbpc[:, j] .= sbp[:, j] .- sbp[ii, j]
        end

        # Fit a low rank model to the quantiles
        u, v = fit_flr(sbpc, 500., 5000.)
        p = lineplot(u, title=@sprintf("%s BMI loadings at age %.0f", sex, age))
        println(p)
        p = lineplot(v, title=@sprintf("%s SBP loadings at age %.0f", sex, age))
        println(p)

        # Calculate the R^2
        r1f = u * v'
        resid = sbpc - r1f
        println("R^2=", sum(abs2, r1f) / sum(abs2, sbpc))

    end

    println(plt0)
    println(plt1)
    println(plt2)

end
