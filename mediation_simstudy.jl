using Distributions, PyPlot, Printf, Statistics, UnicodePlots

rm("plots", force = true, recursive = true)
mkdir("plots")

include("mediation.jl")
include("functional_lr.jl")

n = 2000
m = 11

pg = collect(range(1 / m, 1 - 1 / m, length = m))

age1 = 0.1 * randn(n)
age2 = 0.1 * randn(n)

x = randn(n)
m1 = x + randn(n)
m2 = x + randn(n)
y = x + m1 + m2 + randn(n)
xm = hcat(age1, age2, x, m1, m2)

exq = quantile(Normal(0, 1), pg)
m1q = quantile(Normal(0, sqrt(2)), pg)
m2q = quantile(Normal(0, sqrt(2)), pg)

bw = [1.0, 1, 1, 1, 1]

xr = mediation_quantiles(y, xm, pg, exq, m1q, m2q, bw)

function make_plots(ifig::Int)::Int
    for j = 1:m

        # What we want
        x = exq .+ quantile(Normal(), pg[j])

        # What we have
        y = xr[:, 6, 6, j]

        PyPlot.clf()
        PyPlot.grid(true)
        PyPlot.plot(x, y, "-", color = "orange")
        PyPlot.plot([-3, 3], [-3, 3], "-", color = "black")
        PyPlot.xlabel("Actual quantiles", size = 15)
        PyPlot.ylabel("Estimated quantiles", size = 15)
        PyPlot.title(@sprintf("%.2f quantile of y given x, M1=0, M2=0", pg[j]))
        PyPlot.savefig(@sprintf("plots/%03d.pdf", ifig))
        ifig += 1

    end

    return ifig
end

ifig = make_plots(0)

function make_plot2(ifig)

    PyPlot.clf()
    PyPlot.grid(true)

    qx = quantile(Normal(0, 1), pg)
    qm = quantile(Normal(0, sqrt(2)), pg)
    slopes, icepts = Float64[], Float64[]
    for i1 = 1:m
        for i2 = 1:m
            for i3 = 1:m
                y = xr[i1, i2, i3, :]
                x = qx .+ (qx[i1] + qm[i2] + qm[i3])
                slope = cov(y, x) / var(x)
                icept = mean(y) - slope * mean(x)
                push!(slopes, slope)
                push!(icepts, icept)
                PyPlot.plot(x, y, "-", alpha = 0.3, color = "grey")
            end
        end
    end
    PyPlot.xlabel("Actual quantiles", size = 15)
    PyPlot.ylabel("Estimated quantiles", size = 15)
    PyPlot.savefig(@sprintf("plots/%03d.pdf", ifig))
    ifig += 1

    PyPlot.clf()
    PyPlot.grid(true)
    PyPlot.hist(icepts)
    PyPlot.xlabel("Quantile function intercept", size = 15)
    PyPlot.ylabel("Frequency", size = 15)
    PyPlot.savefig(@sprintf("plots/%03d.pdf", ifig))
    ifig += 1

    PyPlot.clf()
    PyPlot.grid(true)
    PyPlot.hist(slopes)
    PyPlot.xlabel("Quantile function slope", size = 15)
    PyPlot.ylabel("Frequency", size = 15)
    PyPlot.xlim(xmin = 0)
    PyPlot.savefig(@sprintf("plots/%03d.pdf", ifig))
    ifig += 1

    return ifig
end

ifig = make_plot2(ifig)

# Fit a low rank model to the estimated quantiles
u, v = fit_flr_tensor(vec(xr), m, 4, fill(1.0, 3), fill(1.0, 3))

# Put the fitted low rank model into interpretable coordinates
mn, uu, vv = reparameterize(u, v)

# Fitted fitted values from the low-rank fit
function check_fit(ifig)
    xrf1 = zeros(m, m, m, m)
    xrf2 = zeros(m, m, m, m)
    qt = zeros(m, m, m, m)
    qx = quantile(Normal(0, 1), pg)
    qm = quantile(Normal(0, sqrt(2)), pg)
    for i1 = 1:m
        for i2 = 1:m
            for i3 = 1:m
                for j = 1:m

                    # The true quantile
                    qt[i1, i2, i3, j] = qx[i1] + qm[i2] + qm[i3] + qx[j]

                    # Fitted quantiles
                    f = mn[j]
                    f += uu[i1, 1] * vv[j, 1]
                    f += uu[i2, 2] * vv[j, 2]
                    f += uu[i3, 3] * vv[j, 3]
                    xrf1[i1, i2, i3, j] = f
                    xrf2[i1, i2, i3, j] = u[i1, 1] * v[j, 1]
                    xrf2[i1, i2, i3, j] += u[i2, 2] * v[j, 2]
                    xrf2[i1, i2, i3, j] += u[i3, 3] * v[j, 3]
                end
            end
        end
    end

    # Check that fitted values under two parameterizations agree
    @assert maximum(abs.(xrf1 - xrf2)) < 1e-8

    # Plot low-rank fitted versus raw fitted quantiles
    PyPlot.clf()
    PyPlot.grid(true)
    PyPlot.plot(vec(xr), vec(xrf1), ".", color = "grey", alpha = 0.2, rasterized = true)
    PyPlot.xlabel("Raw fitted quantiles", size = 15)
    PyPlot.ylabel("Low-rank fitted quantiles", size = 15)
    PyPlot.savefig(@sprintf("plots/%03d.pdf", ifig))
    ifig += 1

    # Plot low-rank fitted versus true quantiles
    PyPlot.clf()
    PyPlot.grid(true)
    PyPlot.plot(vec(qt), vec(xrf1), ".", color = "grey", alpha = 0.2, rasterized = true)
    PyPlot.xlabel("True quantiles", size = 15)
    PyPlot.ylabel("Low-rank fitted quantiles", size = 15)
    PyPlot.savefig(@sprintf("plots/%03d.pdf", ifig))
    ifig += 1

    # Plot raw fitted versus true quantiles
    PyPlot.clf()
    PyPlot.grid(true)
    PyPlot.plot(vec(qt), vec(xr), ".", color = "grey", alpha = 0.2, rasterized = true)
    PyPlot.xlabel("True quantiles", size = 15)
    PyPlot.ylabel("Raw fitted quantiles", size = 15)
    PyPlot.savefig(@sprintf("plots/%03d.pdf", ifig))
    ifig += 1

    # Histogram of estimation errors from raw fit
    PyPlot.clf()
    PyPlot.grid(true)
    PyPlot.hist(vec(xr) - vec(qt))
    PyPlot.xlabel("Raw fit estimation residual", size = 15)
    PyPlot.ylabel("Frequency", size = 15)
    PyPlot.savefig(@sprintf("plots/%03d.pdf", ifig))
    ifig += 1

    # Histogram of estimation errors from low-rank fit
    PyPlot.clf()
    PyPlot.grid(true)
    PyPlot.hist(vec(xrf1) - vec(qt))
    PyPlot.xlabel("Low-rank fit estimation residual", size = 15)
    PyPlot.ylabel("Frequency", size = 15)
    PyPlot.savefig(@sprintf("plots/%03d.pdf", ifig))
    ifig += 1

    # Check that the fitted values are close to the truth
    println(cor(vec(xr), vec(xrf1)))
    println(mean(abs.(vec(xr) - vec(xrf1))))

    return ifig
end

ifig = check_fit(ifig)

f = [@sprintf("plots/%03d.pdf", j) for j = 0:ifig-1]
c = `gs -sDEVICE=pdfwrite -dNOPAUSE -dBATCH -dSAFER -sOutputFile=mediation_simstudy.pdf $f`
run(c)
