using DataFrames, CSV, GZip, ProcessRegression, UnicodePlots, Statistics, Optim
using PyPlot, Printf, LinearAlgebra

rm("plots", recursive=true, force=true)
mkdir("plots")

df = GZip.open("/home/kshedden/data/Beverly_Strassmann/Cohort_2021.csv.gz") do io
    CSV.read(io, DataFrame)
end

m1var = :Ht_Ave_Use
m2var = :BMI
yvar = :SBP_MEAN

trans = Dict(:Ht_Ave_Use=>"Height", :BMI=>"BMI", :SBP_MEAN=>"SBP")

# Spline-like basis for mean functions
function get_basis(nb, age; scale=4)
	xb = zeros(length(age), nb)
    age1 = minimum(skipmissing(age)) 
    age2 = maximum(skipmissing(age))
    for (j, v) in enumerate(range(age1, age2, length=nb))
	    u = (age .- v) ./ scale
	    xb[:, j] = exp.(-u.^2 ./ 2)
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
    dx = filter(r->r.Sex == sex, df)
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

function get_meanmat(mode, dx, age, nb)
	x = ones(size(dx, 1), [nb, nb+1, nb+2][mode])
	x[:, 1:nb] = get_basis(nb, age)
	if mode >= 2
		x[:, nb+1] = dx[:, m1var]
	elseif mode >= 3
		x[:, nb+2] = dx[:, m2var]
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

function fitmodel(mode, sex; nb=5)
    dx = get_dataframe(mode, sex)
    age = Vector{Float64}(dx[:, :Age_Yrs])
    xb = get_basis(nb, age)

    response = get_response(mode, dx)
    response = (response .- mean(response)) ./std(response)
    meanmat = get_meanmat(mode, dx, age, nb)
    scalemat = get_scalemat(mode, dx)
    smoothmat = get_smoothmat(mode, dx)
    unexplainedmat = get_unexplainedmat(mode, dx)
    idv = Vector{Int}(dx[:, :ID])
    xm = Xmat(meanmat, scalemat, smoothmat, unexplainedmat)

    pm = ProcessMLEModel(response, xm, age, idv)
    pm.penalty = Penalty(0, 1000, 1000, 0)
    fit!(pm; verbose=true, maxiter=400, maxiter_gd=50, g_tol=1e-4)
	return pm
end

function gencovfig(sex, mode, par, ifg)
	aa = mode == 3 ? range(10, 25, length=50) : range(1, 25, length=50)
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
	    PyPlot.title(sex * " " * trans[[m1var, m2var, yvar][mode]] * (r ? " correlation" : " covariance"))
	    PyPlot.imshow(cm, interpolation="nearest", origin="upper", extent=[1, 25, 25, 1])
	    PyPlot.colorbar()
	    PyPlot.xlabel("Age", size=15)
	    PyPlot.ylabel("Age", size=15)
	    PyPlot.savefig(@sprintf("plots/%03d.pdf", ifg))
	    ifg += 1
	end

	return ifg
end

function main(ifg, sex)
    pma = []
    for mode in [1, 2, 3]
	    px = fitmodel(mode, sex)
        push!(pma, px)
	    ifg = gencovfig(sex, mode, px.params, ifg)
    end
	return ifg, pma
end

ifg = 0
ifg, pma_f = main(ifg, "Female")
ifg, pma_m = main(ifg, "Male")

f = [@sprintf("plots/%03d.pdf", j) for j in 0:ifg-1]
c = `gs -sDEVICE=pdfwrite -dNOPAUSE -dBATCH -dSAFER -sOutputFile=dogon_covariances.pdf $f`
run(c)

#pm.penalty = Penalty(1000, 100, 100, 0)
#fit!(pm, start=pm.params, verbose=true)

error("")
# Fitted mean curve
ff = xb * coef(pm)[1:6]
scatterplot(age, ff)

pme = emulate(pm.params; X=pm.X, grp=pm.grplab, time=pm.time, penalty=penalty)
#pme.fix_unexplained = pm.fix_unexplained

fit!(pme, verbose=true)
