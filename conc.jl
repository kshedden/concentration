using CSV,
    GZip,
    DataFrames,
    QuantileRegressions,
    NamedArrays,
    Interpolations,
    UnicodePlots,
    LinearAlgebra,
    Printf

pa = "/afs/umich.edu/user/k/s/kshedden/Projects/Beverly_Strassmann/ursps21/cohort.csv.gz"

da = GZip.open(pa) do io
    CSV.read(io, DataFrame)
end

function prep(sex)
    dx = filter(x -> !ismissing(x.Sex) && x.Sex == sex, da)
    dx = dx[:, [:Age_Yrs, :BMI, :WT, :Ht_Ave_Use, :SBP_MEAN]]
    dx = dx[completecases(dx), :]
    dx = NamedArray(Array{Float64}(dx))
    setnames!(dx, ["Age", "BMI", "WT", "HT", "SBP"], 2)
    dx
end

# Separate datasets for females and males.
df = prep("Female")
dm = prep("Male")

# Returns an estimate of the quantiles of u given v=x, using the data dx.
# h is a grid parameter that controls how finely the quantile function is
# calculated.
function quant(x, u, v, dx; h = 0.02)
    pl = 0.02:h:0.98 # probability points
    ql = zeros(length(pl))
    for (k, p) in enumerate(pl)
        rr = QuantileRegressions.npqreg(Array(dx[:, u]), Array(dx[:, v]), p)
        itp = interpolate(tuple(rr[1]), rr[2], Gridded(Linear()))
        ql[k] = itp(x)
    end
    tuple(pl, ql)
end

# Returns a function that uses local linear regression to estimate E[Y
# | X], with bandwidths in b.  X should not contain an intercept, as
# that is added internally.
function locreg(y, x, b)

    y, x = Array(y), Array(x)

    # Number of covariates
    p = size(x, 2)

    # Include an intercept
    x1 = ones(size(x, 1), size(x, 2) + 1)
    x1[:, 2:end] = x

    # Storage for weights
    w = zeros(length(y))

    function (z)
        for i in eachindex(y)
            ssq = 0.0
            for j = 1:p
                ssq = ssq + ((x[i, j] - z[j]) / b[j])^2
            end
            w[i] = exp(-ssq / 2)
        end
        dw = Diagonal(w)
        c = (x1' * dw * x1) \ (x1' * dw * y)
        c[1] + sum(z .* c[2:end])
    end
end

# Estimate the concentration curve at the given age, using the data in
# dx.  The returned array has three columns.  The first column is the
# probability points, the second column is the concentration curve
# values, the third column is the conditional means which, when
# integrated, yield the concentration curve.  The bandwidth values in
# bw are for age and the control variable, respectively.  The value of
# h is the mesh on which the quantile function is calculated.
function concentration(out, ctrl, age, dx; bw = [2.0, 2.0], h = 0.02)

    # Estimate the quantiles
    pl, ql = quant(age, ctrl, "Age", dx, h = h)

    # Marginal mean of the outcome given Age=age.
    eym = locreg(dx[:, out], dx[:, "Age"], bw[1])
    m0 = eym(age)

    # Conditional mean function of the outcome
    ey = locreg(dx[:, out], dx[:, ["Age", ctrl]], bw)

    # Calculate the conditional mean at each quantile point of the control
    # variable
    cc = zeros(length(pl), 3)
    for (j, q) in enumerate(ql)
        e = ey([age, q]) - m0
        cc[j, :] = [pl[j], e, e]
    end

    # Integrate to get the concentration curve
    cc[:, 2] .= cumsum(cc[:, 3]) .* h
    cc
end


function run1(out, ctrl, age, dx, bw; h = 0.1)
    ey = locreg(dx[:, out], dx[:, ["Age", ctrl]], bw)
    ex = extrema(dx[:, ctrl])
    cv = range(ex[1], stop = ex[2], length = 20)
    y = [ey([age, b]) for b in cv]

    # Plot the expected values as a function of the control variables value
    p = lineplot(cv, y, xlabel = ctrl, ylabel = out, title = @sprintf("Age=%.0f", age))
    println(p)

    cc = concentration(out, ctrl, age, dx, h = h, bw = bw)

    # Plot the expected values as a function of the control variables quantiles
    p = lineplot(
        cc[:, 1],
        cc[:, 3],
        xlabel = "Probability",
        ylabel = @sprintf("Mean %s", out),
        title = @sprintf("Age=%.0f", age),
    )
    println(p)

    # Plot the concentration function
    p = lineplot(
        cc[:, 1],
        cc[:, 2],
        xlabel = "Probability",
        ylabel = "Concentration",
        title = @sprintf("Age=%.0f", age),
    )
    println(p)
end


run1("SBP", "BMI", 18, df, [2.0, 2.0])
run1("SBP", "HT", 18, df, [2.0, 5.0])
run1("SBP", "WT", 18, df, [2.0, 5.0])
