using DataFrames, CSV, GZip, ProcessRegression, UnicodePlots, Statistics, Optim
using PyPlot, Printf, LinearAlgebra, UnicodePlots

rm("plots", recursive = true, force = true)
mkdir("plots")

df = GZip.open("/home/kshedden/data/Beverly_Strassmann/Cohort_2021.csv.gz") do io
    CSV.read(io, DataFrame)
end

maxiter = 500
maxiter_gd = 50

m1var = :Ht_Ave_Use
m2var = :BMI
yvar = :SBP_MEAN

trans = Dict(:Ht_Ave_Use => "Height", :BMI => "BMI", :SBP_MEAN => "SBP")

# Spline-like basis for mean functions
function get_basis(nb, age; scale = 4)
    xb = ones(length(age), nb)
    age1 = minimum(skipmissing(age))
    age2 = maximum(skipmissing(age))
    for (j, v) in enumerate(range(age1, age2, length = nb - 1))
        u = (age .- v) ./ scale
        v = exp.(-u .^ 2 ./ 2)
        xb[:, j+1] = v .- mean(v)
    end
    return xb
end

# mode=1 samples m1
# mode=2 samples m2 | m1
# mode=3 samples y | m1, m2
function get_dataframe(mode, sex)
    @assert sex in ["Female", "Male"]
    vn = [:ID, :Age_Yrs, :Sex, m1var]
    if mode >= 2
        push!(vn, m2var)
    end
    if mode == 3
        push!(vn, yvar)
    end
    dx = filter(r -> r.Sex == sex, df)
    dx = dx[:, vn]
    dx = dx[completecases(dx), :]
    return dx
end

function get_response(mode, dx)
    y = if mode == 1
        dx[:, m1var]
    elseif mode == 2
        dx[:, m2var]
    else
        dx[:, yvar]
    end
    return Vector{Float64}(y)
end

function get_meanmat(mode, age, nb; m1dat = nothing, m2dat = nothing)
    xb = get_basis(nb, age)
    x = if mode == 1
        xb
    elseif mode == 2
        hcat(xb, xb .* m1dat)
    elseif mode == 3
        hcat(xb, xb .* m1dat, xb .* m2dat)
    else
        error("!!")
    end
    return x
end

function get_scalemat(mode, dx)
    x = ones(size(dx, 1), 2)
    x[:, 2] = dx[:, :Age_Yrs]
    return x
end

function get_smoothmat(mode, dx)
    return get_scalemat(mode, dx)
end

function get_unexplainedmat(mode, dx)
    return ones(size(dx, 1), 1)
end

function gen_penalty(mode, age1::Float64, age2::Float64, np::Int, nb::Int)
    age = collect(range(age1, age2, length = np))
    xb = get_basis(nb, age)
    f2 = zeros(np - 2, np)
    for i = 1:np-2
        f2[i, i:i+2] = [1, -2, 1]
    end
    md = f2 * xb
    md2 = md' * md
    qq = zeros(nb * mode, nb * mode)
    for i = 1:mode
        ii = (i - 1) * nb
        qq[ii+1:ii+nb, ii+1:ii+nb] = md2
    end

    return qq
end

function plot_basis(xb, age, ifg)
    ii = sortperm(age)
    PyPlot.clf()
    PyPlot.grid(true)
    for j = 1:size(xb, 2)
        PyPlot.plot(age[ii], xb[ii, j], "-")
    end
    PyPlot.savefig(@sprintf("plots/%03d.pdf", ifg))
    return ifg + 1
end

function fitmodel(mode, sex, ifg; nb = 5)
    dx = get_dataframe(mode, sex)
    age = Vector{Float64}(dx[:, :Age_Yrs])
    xb = get_basis(nb, age)
    ifg = plot_basis(xb, age, ifg)

    others = if mode == 1
        NamedTuple()
    elseif mode == 2
        (m1dat = dx[:, m1var],)
    elseif mode == 3
        (m1dat = dx[:, m1var], m2dat = dx[:, m2var])
    end

    response = get_response(mode, dx)
    meanmat = get_meanmat(mode, age, nb; others...)
    scalemat = get_scalemat(mode, dx)
    smoothmat = get_smoothmat(mode, dx)
    unexplainedmat = get_unexplainedmat(mode, dx)
    idv = Vector{Int}(dx[:, :ID])
    xm = Xmat(meanmat, scalemat, smoothmat, unexplainedmat)

    f = 1e5
    pen = Penalty(
        f * gen_penalty(mode, 1.0, 25.0, 100, nb),
        zeros(0, 0),
        zeros(0, 0),
        zeros(0, 0),
    )
    pm = ProcessMLEModel(response, xm, age, idv; penalty = pen)
    fit!(pm; verbose = false, maxiter = maxiter, maxiter_gd = maxiter_gd, 
         g_tol = 1e-4, skip_se = true)

    return pm, ifg
end

function genmeanfig(sex, mode, nb, pma, ifg)
    age1 = collect(range(1, 25, length = 50))
    age2 = collect(range(10, 25, length = 50))

    # Expected value of m1var
    meanmat1 = get_meanmat(1, age1, nb)
    e1 = meanmat1 * pma[1].params.mean

    if mode == 1
        PyPlot.clf()
        PyPlot.grid(true)
        PyPlot.plot(age1, e1, "-")
        PyPlot.xlabel("Age")
        PyPlot.ylabel(@sprintf("%s %s", sex, trans[m1var]))
        PyPlot.savefig(@sprintf("plots/%03d.pdf", ifg))
        return ifg + 1
    end

    # Expected value of m2var
    e2 = []
    fv = [0.8, 1.0, 1.2]
    for f in fv
        meanmat2 = get_meanmat(2, age1, nb; m1dat = f * e1)
        push!(e2, meanmat2 * pma[2].params.mean)
    end

    if mode == 2
        PyPlot.clf()
        PyPlot.figure(figsize = (8, 5))
        PyPlot.axes([0.1, 0.1, 0.7, 0.8])
        PyPlot.grid(true)
        for (j, f) in enumerate(fv)
            PyPlot.plot(age1, e2[j], "-", label = @sprintf("%.1f", f))
        end
        ha, lb = PyPlot.gca().get_legend_handles_labels()
        leg = PyPlot.figlegend(ha, lb, "center right")
        leg.draw_frame(false)
        leg.set_title(trans[m1var])
        PyPlot.xlabel("Age")
        PyPlot.ylabel(@sprintf("%s %s", sex, trans[m2var]))
        PyPlot.savefig(@sprintf("plots/%03d.pdf", ifg))
        return ifg + 1
    end

    # Expected value of yvar
    e3, ft = [], []
    for (j1, f1) in enumerate(fv)
        for (j2, f2) in enumerate(fv)
            meanmat1 = get_meanmat(1, age2, nb)
            e1 = meanmat1 * pma[1].params.mean
            meanmat2 = get_meanmat(2, age2, nb; m1dat = f1 * e1)
            e2 = meanmat2 * pma[2].params.mean
            meanmat3 = get_meanmat(3, age2, nb; m1dat = f1 * e1, m2dat = f2 * e2)
            push!(e3, meanmat3 * pma[3].params.mean)
            push!(ft, [f1, f2])
        end
    end

    PyPlot.clf()
    PyPlot.figure(figsize = (8, 5))
    PyPlot.axes([0.1, 0.1, 0.7, 0.8])
    PyPlot.grid(true)
    for j = 1:length(e3)
        PyPlot.plot(age2, e3[j], "-", label = @sprintf("%.1f/%.1f", ft[j][1], ft[j][2]))
    end
    ha, lb = PyPlot.gca().get_legend_handles_labels()
    leg = PyPlot.figlegend(ha, lb, "center right")
    leg.draw_frame(false)
    leg.set_title(@sprintf("%s/%s", trans[m1var], trans[m2var]))
    PyPlot.xlabel("Age")
    PyPlot.ylabel(@sprintf("%s %s", sex, trans[yvar]))
    PyPlot.savefig(@sprintf("plots/%03d.pdf", ifg))
    return ifg + 1
end

function gencovfig(sex, mode, par, ifg)
    aa = mode == 3 ? range(10, 25, length = 50) : range(1, 25, length = 50)
    aa = collect(aa)
    x = ones(length(aa), 2)
    x[:, 2] = aa
    lsc = x * par.scale
    lsm = x * par.smooth
    lux = x[:, 1:1] * par.unexplained
    cpar = GaussianCovPar(lsc, lsm, lux)
    cm = covmat(cpar, aa)

    for r in [false, true]
        if r
            s = sqrt.(diag(cm))
            cm ./= s * s'
        end
        PyPlot.clf()
        PyPlot.title(
            sex *
            " " *
            trans[[m1var, m2var, yvar][mode]] *
            (r ? " correlation" : " covariance"),
        )
        PyPlot.imshow(
            cm,
            interpolation = "nearest",
            origin = "upper",
            extent = [1, 25, 25, 1],
        )
        PyPlot.colorbar()
        PyPlot.xlabel("Age", size = 15)
        PyPlot.ylabel("Age", size = 15)
        PyPlot.savefig(@sprintf("plots/%03d.pdf", ifg))
        ifg += 1
    end

    return ifg
end

function plot_emulated(mode, em, par, px, ifg)

	y_mean = px.X.mean * par.mean

	# Plot a limited number of subjects
	ii = 1
	for k in 1:10
        # Don't plot people with few time points
		while px.grp[2, ii] - px.grp[1, ii] < 4
		    ii += 1
		end
		i1, i2 = px.grp[:, ii]
		ti = px.time[i1:i2]
		PyPlot.clf()
		PyPlot.axes([0.1, 0.1, 0.7, 0.8])
		PyPlot.grid(true)
		PyPlot.title(@sprintf("Subject %d", ii))
		PyPlot.xlabel("Age", size=15)
		PyPlot.ylabel(trans[[m1var, m2var, yvar][mode]], size=15)
		PyPlot.plot(ti, y_mean[i1:i2], "-", color="black", label="Mean")
		PyPlot.plot(ti, px.y[i1:i2], "-", color="orange", label="Observed")
		for e in em
			PyPlot.plot(ti, e[i1:i2], "-", color="grey", alpha=0.5, 
			            label="Simulated")
		end
		ha, lb = plt.gca().get_legend_handles_labels()
		leg = PyPlot.figlegend(ha[1:3], lb[1:3], "center right")
		leg.draw_frame(false)
		PyPlot.savefig(@sprintf("plots/%03d.pdf", ifg))
		ii += 1
		ifg += 1
	end

	return ifg
end

function main(ifg, sex)
    pma = []
    nb = 5
    for mode in [1, 2, 3]
        px, ifg = fitmodel(mode, sex, ifg; nb = nb)
        push!(pma, px)
        ifg = genmeanfig(sex, mode, nb, pma, ifg)
        ifg = gencovfig(sex, mode, px.params, ifg)

		em = [emulate(px) for j in 1:10]
		ifg = plot_emulated(mode, em, px.params, px, ifg)
    end
    return ifg, pma
end

ifg = 0
ifg, pma_f = main(ifg, "Female")
ifg, pma_m = main(ifg, "Male")

f = [@sprintf("plots/%03d.pdf", j) for j = 0:ifg-1]
c = `gs -sDEVICE=pdfwrite -dNOPAUSE -dBATCH -dSAFER -sOutputFile=dogon_procmodel.pdf $f`
run(c)
