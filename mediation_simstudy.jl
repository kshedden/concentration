using Distributions, PyPlot, Printf, Statistics, UnicodePlots

rm("plots", force = true, recursive = true)
mkdir("plots")

include("mediation.jl")
include("functional_lr.jl")

n = 1000
m = 11
pg = collect(range(1 / m, 1 - 1 / m, length = m))

age1 = 0.1 * randn(n)
age2 = 0.1 * randn(n)

# Generate the exposure
exs = 1.0
ex = exs .* randn(n)

# Generate the first mediator
m1a, m1b = 2.0, 3.0
m1 = m1a .* ex + m1b .* randn(n)
m1s = sqrt(m1a^2 * exs^2 + m1b^2)

# Generate the second mediator
m2a, m2b = 3.0, 2.0
m2 = m2a .* ex + m2b .* randn(n)
m2s = sqrt(m2a^2 * exs^2 + m2b^2)

# Generate the response
yb, y1b, y2b, ys = 2.0, 3.0, 4.0, 5.0
yv = yb .* ex + y1b .* m1 + y2b .* m2 + ys .* randn(n)

# Put everything into a design matrix
xm = hcat(age1, age2, ex, m1, m2)

# Marginal quantiles of the exposures and mediators
exq = quantile(Normal(0, exs), pg)
m1q = quantile(Normal(0, m1s), pg)
m2q = quantile(Normal(0, m2s), pg)

bw = [1.0, 1, 1, 1, 1]

# Compute conditional quantiles of y given x, m1, and m2.
xr = mediation_quantiles(yv, xm, pg, exq, m1q, m2q, bw)

# Remove the median
xc, md = center(xr)

function make_plot0(ifig::Int)::Int
    PyPlot.clf()
    PyPlot.grid(true)
    x = quantile(Normal(0, ys), pg)
    PyPlot.plot(x, md, "-", color = "orange")
    PyPlot.plot([minimum(x), maximum(x)], [minimum(md), maximum(md)], "-", color = "grey")
    PyPlot.xlabel("Actual quantiles", size = 15)
    PyPlot.ylabel("Estimated quantiles", size = 15)
    PyPlot.title("Quantiles at median exposure and predictors")
    PyPlot.savefig(@sprintf("plots/%03d.pdf", ifig))
    return ifig + 1
end

# Plot the y quantiles as a function of x, when both mediators are fixed at their median.
function make_plots(ifig::Int)::Int
    for j = 1:m
        # What we want
        x = yb .* exq .+ quantile(Normal(0, ys), pg[j])

        # What we have
        y = xr[:, 6, 6, j]

        PyPlot.clf()
        PyPlot.grid(true)
        PyPlot.plot(x, y, "-", color = "orange")
        PyPlot.plot(
            [minimum(x), maximum(x)],
            [minimum(y), maximum(y)],
            "-",
            color = "black",
        )
        PyPlot.xlabel("Actual quantiles", size = 15)
        PyPlot.ylabel("Estimated quantiles", size = 15)
        PyPlot.title(@sprintf("%.2f quantile of y given x, M1=0, M2=0", pg[j]))
        PyPlot.savefig(@sprintf("plots/%03d.pdf", ifig))
        ifig += 1
    end

    return ifig
end


function make_plot2(ifig)

    PyPlot.clf()
    PyPlot.grid(true)

    qy = quantile(Normal(0, ys), pg)
    slopes, icepts = Float64[], Float64[]
    for i1 = 1:m
        for i2 = 1:m
            for i3 = 1:m
                y = xr[i1, i2, i3, :]
                x = qy .+ (yb .* exq[i1] + y1b .* m1q[i2] + y2b .* m2q[i3])
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

# Fit a low rank model to the estimated quantiles
u, v = fit_flr_tensor(xc, fill(1.0, 3), fill(1.0, 3))

# Fitted fitted values from the low-rank fit
function check_fit(ifig)
    xrf = getfit(u, v)
    qtc = zeros(m, m, m, m)
    qy = quantile(Normal(0, ys), pg)
    for i1 = 1:m
        for i2 = 1:m
            for i3 = 1:m
                for j = 1:m
                    # The true quantile after centering
                    qtc[i1, i2, i3, j] = yb .* exq[i1] + y1b .* m1q[i2] + y2b .* m2q[i3]
                end
            end
        end
    end

    # Plot low-rank fitted versus raw fitted quantiles
    PyPlot.clf()
    PyPlot.grid(true)
    PyPlot.plot(vec(xc), vec(xrf), ".", color = "grey", alpha = 0.2, rasterized = true)
    PyPlot.xlabel("Centered raw fitted quantiles", size = 15)
    PyPlot.ylabel("Centered low-rank fitted quantiles", size = 15)
    PyPlot.savefig(@sprintf("plots/%03d.pdf", ifig))
    ifig += 1

    # Plot low-rank fitted versus true quantiles
    PyPlot.clf()
    PyPlot.grid(true)
    PyPlot.plot(vec(qtc), vec(xrf), ".", color = "grey", alpha = 0.2, rasterized = true)
    PyPlot.xlabel("Centered true quantiles", size = 15)
    PyPlot.ylabel("Centered low-rank fitted quantiles", size = 15)
    PyPlot.savefig(@sprintf("plots/%03d.pdf", ifig))
    ifig += 1

    # Plot raw fitted versus true quantiles
    PyPlot.clf()
    PyPlot.grid(true)
    PyPlot.plot(vec(qtc), vec(xc), ".", color = "grey", alpha = 0.2, rasterized = true)
    PyPlot.xlabel("Centered true quantiles", size = 15)
    PyPlot.ylabel("Centered raw fitted quantiles", size = 15)
    PyPlot.savefig(@sprintf("plots/%03d.pdf", ifig))
    ifig += 1

    # Histogram of estimation errors from raw fit
    PyPlot.clf()
    PyPlot.grid(true)
    PyPlot.hist(vec(xc) - vec(qtc))
    PyPlot.xlabel("Raw fit estimation residual", size = 15)
    PyPlot.ylabel("Frequency", size = 15)
    PyPlot.savefig(@sprintf("plots/%03d.pdf", ifig))
    ifig += 1

    # Histogram of estimation errors from low-rank fit
    PyPlot.clf()
    PyPlot.grid(true)
    PyPlot.hist(vec(xrf) - vec(qtc))
    PyPlot.xlabel("Low-rank fit estimation residual", size = 15)
    PyPlot.ylabel("Frequency", size = 15)
    PyPlot.savefig(@sprintf("plots/%03d.pdf", ifig))
    ifig += 1

    return ifig
end

ifig = make_plot0(0)
ifig = make_plots(ifig)
ifig = make_plot2(ifig)
ifig = check_fit(ifig)

f = [@sprintf("plots/%03d.pdf", j) for j = 0:ifig-1]
c = `gs -sDEVICE=pdfwrite -dNOPAUSE -dBATCH -dSAFER -sOutputFile=mediation_simstudy.pdf $f`
run(c)
