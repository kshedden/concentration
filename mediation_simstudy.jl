using Distributions, PyPlot, Printf, Statistics, UnicodePlots

rm("plots", force = true, recursive = true)
mkdir("plots")

include("mediation.jl")
include("functional_lr.jl")

n = 4000
m = 5
pg = collect(range(1 / m, 1 - 1 / m, length = m))

mutable struct gspec
    exs::Float64  # Marginal standard deviation of x
    bxm1::Float64 # Slope of m1 on x
    sm1::Float64  # Residual standard deviation of m1
    bxm2::Float64 # Slope of m2 on x
    sm2::Float64  # Residual standard deviation of m2
    bxy::Float64  # Slope of y on x
    bm1y::Float64 # Slope of y on m1
    bm2y::Float64 # Slope of y on m2
    sy::Float64   # Residual standard deviation of y
end

function gendat(n, gs)

    age1 = 0.1 * randn(n)
    age2 = 0.1 * randn(n)

    # Generate the exposure
    ex = gs.exs .* randn(n)

    # Generate the first mediator
    m1 = gs.bxm1 .* ex + gs.sm1 .* randn(n)
    m1s = sqrt(gs.bxm1^2 * gs.exs^2 + gs.sm1^2)

    # Generate the second mediator
    m2 = gs.bxm2 .* ex + gs.sm2 .* randn(n)
    m2s = sqrt(gs.bxm2^2 * gs.exs^2 + gs.sm2^2)

    # Generate the response
    yv = gs.bxy .* ex + gs.bm1y .* m1 + gs.bm2y .* m2 + gs.sy .* randn(n)

    # Put everything into a design matrix
    xm = hcat(age1, age2, ex, m1, m2)

    return yv, xm, m1s, m2s
end

# Check for bias in the central axis
function check_central_axis(ifig::Int)::Int

    # Generate data
    exs = 1.0
    bxm1, sm1 = 2.0, 3.0
    bxm2, sm2 = 3.0, 2.0
    bxy, bm1y, bm2y, sy = 2.0, 3.0, 4.0, 0.5
    gs = gspec(exs, bxm1, sm1, bxm2, sm2, bxy, bm1y, bm2y, sy)
    bw = 0.2 .* [1.0, 1, 1, 1, 1]
    nrep = 20
    mm = zeros(m)

    for j = 1:nrep
        yv, xm, m1s, m2s = gendat(n, gs)

        # Marginal quantiles of the exposures and mediators
        exq = quantile(Normal(0, exs), pg)
        m1q = quantile(Normal(0, m1s), pg)
        m2q = quantile(Normal(0, m2s), pg)

        # Compute conditional quantiles of y given x, m1, and m2.
        xr = mediation_quantiles(yv, xm, pg, exq, m1q, m2q, bw)

        # Remove the central axis
        _, md = center(xr)
        mm .+= md
    end

    mm ./= nrep

    PyPlot.clf()
    PyPlot.grid(true)
    x = quantile(Normal(0, sy), pg)
    PyPlot.plot(x, mm, "-", color = "orange")
    mn = min(minimum(x), minimum(mm))
    mx = max(maximum(x), maximum(mm))
    PyPlot.plot([mn, mx], [mn, mx], "-", color = "black")
    PyPlot.xlabel("Actual quantiles", size = 15)
    PyPlot.ylabel("Estimated quantiles", size = 15)
    PyPlot.title("Quantiles at median exposure and predictors")
    PyPlot.savefig(@sprintf("plots/%03d.pdf", ifig))
    return ifig + 1
end

# Plot the each y quantile as a function of x, when both mediators are fixed at their median.
function check_y_given_x(ifig::Int)::Int

    # Generate data
    exs = 1.0
    bxm1, sm1 = 2.0, 3.0
    bxm2, sm2 = 3.0, 2.0
    bxy, bm1y, bm2y, sy = 2.0, 3.0, 4.0, 0.5
    gs = gspec(exs, bxm1, sm1, bxm2, sm2, bxy, bm1y, bm2y, sy)
    bw = 0.2 .* [1.0, 1, 1, 1, 1]

    yv, xm, m1s, m2s = gendat(n, gs)

    # Marginal quantiles of the exposures and mediators
    exq = quantile(Normal(0, exs), pg)
    m1q = quantile(Normal(0, m1s), pg)
    m2q = quantile(Normal(0, m2s), pg)

    # Compute conditional quantiles of y given x, m1, and m2.
    xr = mediation_quantiles(yv, xm, pg, exq, m1q, m2q, bw)

    for j = 1:m
        # What we want
        x = bxy .* exq .+ quantile(Normal(0, sy), pg[j])

        # What we have
        y = xr[:, div(m + 1, 2), div(m + 1, 2), j]

        PyPlot.clf()
        PyPlot.grid(true)
        PyPlot.plot(x, y, "-", color = "orange")
        mn = min(minimum(x), minimum(y))
        mx = max(maximum(x), maximum(y))
        PyPlot.plot([mn, mx], [mn, mx], "-", color = "black")
        PyPlot.xlabel("Actual quantiles", size = 15)
        PyPlot.ylabel("Estimated quantiles", size = 15)
        PyPlot.title(@sprintf("%.2f quantile of y given x, M1=0, M2=0", pg[j]))
        PyPlot.savefig(@sprintf("plots/%03d.pdf", ifig))
        ifig += 1
    end

    return ifig
end


function check_yquant(ifig)

    # Generate data
    exs = 1.0
    bxm1, sm1 = 2.0, 3.0
    bxm2, sm2 = 3.0, 2.0
    bxy, bm1y, bm2y, sy = 2.0, 3.0, 4.0, 0.5
    gs = gspec(exs, bxm1, sm1, bxm2, sm2, bxy, bm1y, bm2y, sy)
    bw = 0.2 .* [1.0, 1, 1, 1, 1]

    yv, xm, m1s, m2s = gendat(n, gs)

    # Marginal quantiles of the exposures and mediators
    exq = quantile(Normal(0, exs), pg)
    m1q = quantile(Normal(0, m1s), pg)
    m2q = quantile(Normal(0, m2s), pg)

    # Compute conditional quantiles of y given x, m1, and m2.
    xr = mediation_quantiles(yv, xm, pg, exq, m1q, m2q, bw)

    PyPlot.clf()
    PyPlot.grid(true)

    qy = quantile(Normal(0, sy), pg)
    slopes, icepts = Float64[], Float64[]
    for i1 = 1:m
        for i2 = 1:m
            for i3 = 1:m
                y = xr[i1, i2, i3, :]
                mu = bxy .* exq[i1] + bm1y .* m1q[i2] + bm2y .* m2q[i3]
                x = qy .+ mu
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

function check_uv(ifig)

    exs = 1.0
    bxm1, sm1 = 2.0, 3.0
    bxm2, sm2 = 3.0, 2.0
    bxy, bm1y, bm2y, sy = 2.0, 3.0, 4.0, 5.0
    gs = gspec(exs, bxm1, sm1, bxm2, sm2, bxy, bm1y, bm2y, sy)
    bw = 0.2 .* [1.0, 1, 1, 1, 1]
    nrep = 10
    uu = zeros(m, 3)
    vv = zeros(m, 3)
    mm = zeros(m)

    for rp = 1:nrep
        yv, xm, m1s, m2s = gendat(n, gs)

        # Marginal quantiles of the exposures and mediators
        exq = quantile(Normal(0, exs), pg)
        m1q = quantile(Normal(0, m1s), pg)
        m2q = quantile(Normal(0, m2s), pg)

        # Compute conditional quantiles of y given x, m1, and m2.
        xr = mediation_quantiles(yv, xm, pg, exq, m1q, m2q, bw)

        # Remove the median
        xc, md = center(xr)
        mm .+= md

        # Fit a low rank model to the estimated quantiles
        u, v = fit_flr_tensor(xc, fill(1.0, 3), fill(1.0, 3))

        uu .+= u
        vv .+= v

    end

    uu ./= nrep
    vv ./= nrep
    mm ./= nrep

    _, _, m1s, m2s = gendat(n, gs)
    qx = quantile(Normal(0, exs), pg)
    qm1 = quantile(Normal(0, m1s), pg)
    qm2 = quantile(Normal(0, m2s), pg)

    utrue = hcat(bxy .* qx .* sqrt(m), bm1y .* qm1 .* sqrt(m), bm2y .* qm2 .* sqrt(m))
    vtrue = ones(m, 3) ./ sqrt(m)

    for j = 1:3
        plt.clf()
        plt.axes([0.1, 0.1, 0.7, 0.8])
        plt.grid(true)
        plt.plot(pg, uu[:, j], label = "u$(j)")
        plt.plot(pg, utrue[:, j], label = "True u$(j)")
        plt.xlabel("Probability point", size = 15)
        plt.ylabel("Score", size = 15)
        ha, lb = plt.gca().get_legend_handles_labels()
        leg = plt.figlegend(ha, lb, "center right")
        leg.draw_frame(false)
        PyPlot.savefig(@sprintf("plots/%03d.pdf", ifig))
        ifig += 1
    end

    for j = 1:3
        plt.clf()
        plt.axes([0.1, 0.1, 0.7, 0.8])
        plt.grid(true)
        plt.plot(pg, vv[:, j], label = "v$(j)")
        plt.plot(pg, vtrue[:, j], label = "True v$(j)")
        plt.xlabel("Probability point", size = 15)
        plt.ylabel("Loading", size = 15)
        if minimum(vv[:, j]) > 0
            plt.ylim(0, 1.2 * maximum(vv[:, j]))
        elseif maximum(vv[:, j]) < 0
            plt.ylim(1.2 * minimum(vv[:, j]), 0)
        end
        ha, lb = plt.gca().get_legend_handles_labels()
        leg = plt.figlegend(ha, lb, "center right")
        leg.draw_frame(false)
        PyPlot.savefig(@sprintf("plots/%03d.pdf", ifig))
        ifig += 1
    end

    return ifig
end

# Fitted values from the low-rank fit
function check_fit(ifig)

    exs = 1.0
    bxm1, sm1 = 2.0, 3.0
    bxm2, sm2 = 3.0, 2.0
    bxy, bm1y, bm2y, sy = 2.0, 3.0, 4.0, 0.5
    gs = gspec(exs, bxm1, sm1, bxm2, sm2, bxy, bm1y, bm2y, sy)
    bw = 0.2 .* [1.0, 1, 1, 1, 1]

    yv, xm, m1s, m2s = gendat(n, gs)

    # Marginal quantiles of the exposures and mediators
    exq = quantile(Normal(0, exs), pg)
    m1q = quantile(Normal(0, m1s), pg)
    m2q = quantile(Normal(0, m2s), pg)

    # Compute conditional quantiles of y given x, m1, and m2.
    xr = mediation_quantiles(yv, xm, pg, exq, m1q, m2q, bw)

    # Remove the median
    xc, md = center(xr)

    # Fit a low rank model to the estimated quantiles
    u, v = fit_flr_tensor(xc, fill(1.0, 3), fill(1.0, 3))

    xrf = getfit(u, v)
    qtc = zeros(m, m, m, m)
    qy = quantile(Normal(0, sy), pg)
    for i1 = 1:m
        for i2 = 1:m
            for i3 = 1:m
                for j = 1:m
                    # The true quantile after centering
                    qtc[i1, i2, i3, j] = bxy .* exq[i1] + bm1y .* m1q[i2] + bm2y .* m2q[i3]
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

ifig = check_central_axis(0)
#ifig = check_y_given_x(ifig)
#ifig = check_yquant(ifig)
#ifig = check_fit(ifig)
ifig = check_uv(ifig)

f = [@sprintf("plots/%03d.pdf", j) for j = 0:ifig-1]
c = `gs -sDEVICE=pdfwrite -dNOPAUSE -dBATCH -dSAFER -sOutputFile=mediation_simstudy.pdf $f`
run(c)
