using DataFrames, CSV, GZip, ProcessRegression, UnicodePlots, Statistics, Optim
using PyPlot, Printf, LinearAlgebra, UnicodePlots

rm("plots", recursive = true, force = true)
mkdir("plots")

df = GZip.open("/home/kshedden/data/Beverly_Strassmann/Cohort_2021.csv.gz") do io
    CSV.read(io, DataFrame)
end

# Control parameters
maxiter = 10 #500
maxiter_gd = 10 #50
skip_se = false

# Variables to analyze
m1var = :Ht_Ave_Use
m2var = :BMI
yvar = :SBP_MEAN

# Prettify names for plotting
trans = Dict(:Ht_Ave_Use => "Height", :BMI => "BMI", :SBP_MEAN => "SBP")

# Spline-like basis for mean functions
function get_basis(nb, age; scale = 4)
    xb = ones(length(age), nb)
    xb[:, 2] = (age - mean(age)) / maximum(age)
    age1 = minimum(skipmissing(age))
    age2 = maximum(skipmissing(age))
    for (j, v) in enumerate(range(age1, age2, length = nb - 2))
        u = (age .- v) ./ scale
        v = exp.(-u .^ 2 ./ 2)
        xb[:, j+2] = v .- mean(v)
    end
    return xb
end

# mode=1 samples m1
# mode=2 samples m2 | m1
# mode=3 samples y | m1, m2
function get_dataframe(mode, sex)
    @assert sex in ["Female", "Male"]
    vn = [:ID, :datecomb, :Age_Yrs, :Sex, m1var]
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

# The penalty matrix is block-diagonal, with the diagonal blocks being the
# second difference.  For mode = 2, 3, the squared second derivative is
# calculated for the age interactions and for the age main effects.
function gen_penalty(mode, age1::Float64, age2::Float64, np::Int, nb::Int)
    age = collect(range(age1, age2, length = np))
    xb = get_basis(nb, age)
    f2 = zeros(np - 2, np)
    for i = 1:np-2
        f2[i, i:i+2] = [1, -2, 1]
    end
    md = f2 * xb
    md2 = md' * md
	_, s, _ = svd(md2)
	md2 /= maximum(s)
    
    qq = zeros(nb * mode, nb * mode)
    for i = 1:mode
        ii = (i - 1) * nb
        qq[ii+1:ii+nb, ii+1:ii+nb] = md2
    end

    return qq
end

# Plot the basis functions.
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

    # Penalize the mean structure for smoothness.
    f = 1e5
    pen = Penalty(
        f * gen_penalty(mode, 1.0, 25.0, 100, nb),
        zeros(0, 0),
        zeros(0, 0),
        zeros(0, 0),
    )
    pm = ProcessMLEModel(response, xm, age, idv; penalty = pen)
    fit!(
        pm;
        verbose = false,
        maxiter = maxiter,
        maxiter_gd = maxiter_gd,
        g_tol = 1e-4,
        skip_se = skip_se,
    )

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
            @sprintf(
                "%s %s %s",
                sex,
                trans[[m1var, m2var, yvar][mode]],
                r ? " correlation" : " covariance"
            )
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

# Plot emulated data from each piece of the chained model:
# P(M1), P(M2|M1), P(Y|M1, M2).  The values of M1, M2, and Y
# are sampled independently from these three distributions.
function plot_emulated(mode, em, par, px, ifg)

    y_mean = px.X.mean * par.mean

    # Plot a limited number of subjects
    ii = 1
    for k = 1:10
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
        PyPlot.xlabel("Age", size = 15)
        PyPlot.ylabel(trans[[m1var, m2var, yvar][mode]], size = 15)
        PyPlot.plot(ti, y_mean[i1:i2], "-", color = "black", label = "Mean")
        PyPlot.plot(ti, px.y[i1:i2], "-", color = "orange", label = "Observed")
        for e in em
            PyPlot.plot(ti, e[i1:i2], "-", color = "grey", alpha = 0.5, label = "Simulated")
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

function plot_chained_helper(em, id, vn, ifg)
    PyPlot.clf()
    PyPlot.grid(true)
    PyPlot.xlabel("Age", size = 15)
    PyPlot.ylabel(trans[vn], size = 15)
    PyPlot.title("Jointly emulated data")
    vx = Symbol(string(vn) * "_em")
    for idx in id
        for ee in em
            ii = findall(ee[:, :ID] .== idx)
            ex = ee[ii, [:Age_Yrs, vx]]
            ex = ex[completecases(ex), :]
            if size(ex, 1) > 0
                PyPlot.plot(ex[:, :Age_Yrs], ex[:, vx], "-")
            end
        end
    end
    PyPlot.savefig(@sprintf("plots/%03d.pdf", ifg))
    return ifg + 1
end

function plot_chained(pma, em, sex, ifg)

    id = unique(em[1][:, :ID])

    # Find id's to plot.  Don't plot people with very few data points
    idu = []
    j = 1
    while length(idu) < 10
        ii = findall(em[1][:, :ID] .== id[j])
        if length(ii) < 5
            j += 1
            continue
        else
            push!(idu, id[j])
        end
    end

    ifg = plot_chained_helper(em, idu, m1var, ifg)
    ifg = plot_chained_helper(em, idu, m2var, ifg)
    ifg = plot_chained_helper(em, idu, yvar, ifg)

    return ifg
end

# Sample from the joint distribution of (M1, M2, Y) by sampling
# from P(M1), P(M2|M1) and P(Y|M1,M2).  The three models used
# for sampling have been fit ahead of time and stored in the 
# array 'pma'.  The value of 'nb' is the number of basis 
# functions and 'sex' specifies which sex is being analyzed. 
function emulate_chain(pma, nb, sex)

    # Sample M1
    em1 = emulate(pma[1])
    dx1 = get_dataframe(1, sex)
    vn1 = Symbol(string(m1var) * "_em")
    dx1[:, vn1] = em1

    # Sample M2 | M1
    dx2 = get_dataframe(2, sex)
    dx2 = leftjoin(dx2, dx1[:, [:ID, :datecomb, vn1]], on = [:ID, :datecomb])
    age = Vector{Float64}(dx2[:, :Age_Yrs])
    pma[2].X.mean = get_meanmat(2, age, nb; m1dat = dx2[:, vn1])
    em2 = emulate(pma[2])
    vn2 = Symbol(string(m2var) * "_em")
    dx2[:, vn2] = em2

    # Sample Y | M1, M2
    dx3 = get_dataframe(3, sex)
    dx3 = leftjoin(dx3, dx2[:, [:ID, :datecomb, vn1, vn2]], on = [:ID, :datecomb])
    age = Vector{Float64}(dx3[:, :Age_Yrs])
    pma[3].X.mean = get_meanmat(3, age, nb; m1dat = dx3[:, vn1], m2dat = dx3[:, vn2])
    em3 = emulate(pma[3])
    vn3 = Symbol(string(yvar) * "_em")
    dx3[:, vn3] = em3

    dm = outerjoin(
        dx1[:, [:ID, :datecomb, :Age_Yrs, vn1]],
        dx2[:, [:ID, :datecomb, vn2]],
        on = [:ID, :datecomb],
    )
    dm = outerjoin(dm, dx3[:, [:ID, :datecomb, vn3]], on = [:ID, :datecomb])
    return dm
end

function summary_table(pma, fname)
    out = open(fname, "w")
    for px in pma
        s = string(coeftable(px))
        write(out, s)
    end
    close(out)
end

function main(ifg, sex)
    pma = []
    nb = 5
    nrep = 10
    for mode in [1, 2, 3]
        px, ifg = fitmodel(mode, sex, ifg; nb = nb)
        push!(pma, px)
        ifg = genmeanfig(sex, mode, nb, pma, ifg)
        ifg = gencovfig(sex, mode, px.params, ifg)
    end

    # Emulate the observed data for each model.  Any data
    # being conditioned on is fixed at its observed values.
    for mode in [1, 2, 3]
        em = [emulate(pma[mode]) for j = 1:nrep]
        ifg = plot_emulated(mode, em, pma[mode].params, pma[mode], ifg)
    end

    # Emulate the joint distribution of M1, M2, Y by sampling
    # from P(M1), P(M2|M1) and P(Y|M1, M2).  The sampled
    # values from each distribution are passed to the next
    # distribution as conditioning variables.
    em = [emulate_chain(pma, nb, sex) for _ = 1:10]
    ifg = plot_chained(pma, em, sex, ifg)

    summary_table(pma, "procmodels_$(sex).txt")

    return ifg, pma
end

ifg = 0
ifg, pma_f = main(ifg, "Female")
#ifg, pma_m = main(ifg, "Male")

f = [@sprintf("plots/%03d.pdf", j) for j = 0:ifg-1]
c = `gs -sDEVICE=pdfwrite -dNOPAUSE -dBATCH -dSAFER -sOutputFile=dogon_procmodel.pdf $f`
run(c)
