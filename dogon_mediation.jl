using DataFrames, CodecZlib, CSV, Printf, Statistics
using Distributions, UnicodePlots, Interpolations

include("qreg_nn.jl")
include("flr_tensor.jl")
include("dogon_utils.jl")
include("mediation.jl")
include("plot_utils.jl")

rm("plots", recursive = true, force = true)
mkdir("plots")

ifig = 0

df = open(
    GzipDecompressorStream,
    "/home/kshedden/data/Beverly_Strassmann/Cohort_2021.csv.gz",
) do io
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
m = 11

pg = collect(range(1 / m, 1 - 1 / m, length = m))

mr = mediation(qrm; bw = Float64[1, 1, 1])

ifig = plot_tensor(mr.direct, pg, string(cbs), "Direct effect", ifig)
ifig = plot_tensor(mr.indirect1, pg, string(med1), "Indirect effect", ifig)
ifig = plot_tensor(mr.indirect2, pg, string(med2), "Indirect effect", ifig)

f = [@sprintf("plots/%03d.pdf", j) for j = 0:ifig-1]
c = `gs -sDEVICE=pdfwrite -dNOPAUSE -dBATCH -dSAFER -sOutputFile=dogon_mediation.pdf $f`
run(c)
error("")
# Childhood body size variable.
vlc = [vspec(cbs, missing, Inf)]

# Adult body size variables.
vla = [vspec(med1, missing, Inf), vspec(med2, missing, Inf)]

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
    vla;
    cu = 1.0,
    cv = 1.0,
    single = false,
    child_age_caliper = child_age_caliper,
    adult_age_caliper = adult_age_caliper,
)
