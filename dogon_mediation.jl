using DataFrames, GZip, CSV, Printf, Statistics, Distributions, UnicodePlots, Interpolations

include("qreg_nn.jl")
include("functional_lr.jl")
include("dogon_utils.jl")
include("mediation.jl")

df = GZip.open("/home/kshedden/data/Beverly_Strassmann/Cohort_2021.csv.gz") do io
    CSV.read(io, DataFrame)
end

# Analyze one sex, at specific childhood and adult ages
sex = "Female"
age1 = 3.0
age2 = 20.0

# The outcome variable
outcome = :SBP_MEAN

# Childhood body size variable (primary exposure)
# Possibilities: Ht_Ave_Use, WT, HAZ, WAZ
cbs = :BMI

# Mediators
med1 = :Ht_Ave_Use
med2 = :BMI

# Calipers for child and adult age
child_age_caliper = 1.5
adult_age_caliper = 1.5

# Number of quantile points to track
m = 5

pg = collect(range(1 / m, 1 - 1 / m, length = m))

# Childhood body size variable.
vlc = [vspec(cbs, missing, Inf)]

# Adult body size variables.
vla = [vspec(med1, missing, Inf), vspec(med2, missing, Inf)]

# Bandwidth parameters for quantile smoothing
bw = fill(1.0, 5)

qrm = mediation_prep(
    df,
    outcome,
    cbs,
    med1,
    med2,
    sex,
    age1,
    age2,
    m,
    vlc,
    vla,
    bw;
    single = false,
    child_age_caliper = child_age_caliper,
    adult_age_caliper = adult_age_caliper,
)

mr = mediation(qrm)
