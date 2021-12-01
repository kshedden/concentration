using DataFrames, CSV, GZip, ProcessRegression, UnicodePlots

df = GZip.open("/home/kshedden/data/Beverly_Strassmann/Cohort_2021.csv.gz") do io
    CSV.read(io, DataFrame)
end

sex = "Female"

vn = [:ID, :Age_Yrs, :SBP_MEAN, :Sex]

dx = filter(r->r.Sex == sex, df)
dx = dx[:, vn]
dx = dx[completecases(dx), :]

y = Vector{Float64}(dx[:, :SBP_MEAN])
age = Vector{Float64}(dx[:, :Age_Yrs])
idv = Vector{Int}(dx[:, :ID])

# Spline-like basis for mean functions
nb = 5
xb = ones(size(dx, 1), nb+1)
for (j, v) in enumerate(range(minimum(age), maximum(age), length=nb))
	u = (age .- v) ./ 4
	xb[:, j+1] = exp.(-u.^2 ./ 2)
end

x = ones(size(dx, 1), 2)
x[:, 2] = dx[:, :Age_Yrs][:, :]
xm = Xmat(xb, x, x, x)

pm = ProcessMLEModel(y, xm, age, idv)#; fix_unexplained=[1.0, 0])
fit!(pm; verbose=true)

# Fitted mean curve
ff = xb * coef(pm)[1:6]
scatterplot(age, ff)

pme = emulate(pm.params; X=pm.X, grp=pm.grplab, time=pm.time)
#pme.fix_unexplained = pm.fix_unexplained

fit!(pme, verbose=true)
