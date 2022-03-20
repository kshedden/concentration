using DataFrames, CodecZlib, CSV, Printf, Statistics
using Distributions, UnicodePlots, Interpolations

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

# Analyze at specific childhood and adult ages
age1 = 3.0
age2 = 20.0

# The outcome variable
outcome = :SBP_MEAN

# Mediators
med1 = :Ht_Ave_Use
med2 = :BMI

# Calipers for child and adult age
child_age_caliper = 1.5
adult_age_caliper = 3.0

# Number of quantile points to track
m = 11

pg = collect(range(1 / m, 1 - 1 / m, length = m))

# Adult body size variables.
vla = [vspec(med1, missing, Inf), vspec(med2, missing, Inf)]

function main(cbs, sex, ifig)

    # Childhood body size variable.
    vlc = [vspec(cbs, missing, Inf)]

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
            cv = 1000.0,
            single = false,
            child_age_caliper = child_age_caliper,
            adult_age_caliper = adult_age_caliper,
        )

        mr = mediation(qrm; bw = Float64[1, 1, 1])

		title = @sprintf("%s direct effect of %s (age %.0f)\non SBP (age %.0f)", 
		                 sex, string(cbs), age1, age2)
		ylabel = @sprintf("%s (age %.0f)", string(cbs), age1)
        ifig = plot_tensor(mr.dir.effectmap, pg, ylabel, title, ifig)

        title = @sprintf("%s indirect effect of %s (age %.0f)\non SBP (age %.0f) through adult %s", 
                         sex, string(cbs), age1, age2, string(med1)) 
        ylabel = @sprintf("Adult %s", string(med1)) 
        ifig = plot_tensor(mr.indir1.effectmap, pg, ylabel, title, ifig)

		title = @sprintf("%s indirect effect of %s (age %.0f)\non SBP (age %.0f) through adult %s", 
		                 sex, string(cbs), age1, age2, string(med2))
        ylabel = @sprintf("Adult %s", string(med2)) 
        ifig = plot_tensor(mr.indir2.effectmap, pg, ylabel, title, ifig)

    return ifig
end

function main(ifig)
    # Childhood body size variable (primary exposure)
    for cbs in [:WT, :BMI, :Ht_Ave_Use, :HAZ, :BAZ, :WAZ]
		for sex in ["Female", "Male"]
        	ifig = main(cbs, sex, ifig)
        end
    end
    return ifig
end

ifig = 0
ifig = main(ifig)

f = [@sprintf("plots/%03d.pdf", j) for j = 0:ifig-1]
c = `gs -sDEVICE=pdfwrite -dNOPAUSE -dBATCH -dSAFER -sOutputFile=writing/dogon_mediation.pdf $f`
run(c)
