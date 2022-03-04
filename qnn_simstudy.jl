using UnicodePlots, Statistics, Distributions, Printf, LinearAlgebra
using DataFrames, GLM, QuantileNN

#=
We conducted a limited simulation study to assess the bias in the QNN
procedure.  The bias in $\hat{Q}_y(p; x)$ as an estimate of $Q_y(p;
x)$ will be denoted ${\rm bias}(p, x) \equiv E\hat{Q}_y(p; x) - Q_y(p;
x)$. By analogy, the bias of local linear regression depends on the
regularization parameter and neighborood size, the covariate
dimension, and the first and second derivatives of the true mean
function being estimated (ref).  It is likely that these same factors
influence the bias of QNN, but in addition the bias of QNN is also
influenced by the probability point $p$, with bias anticipated to
increase as $p$ moves away from 0.5.

We hypothesized that the bias is largely determined by the difference
between the quantile of interest and the conditional median, $g_p(x)
\equiv Q_y(p; x) - Q_y(0.5; x)$.  To assess this hypothesis, we
conducted a simulation study in which Gaussian data from a
heteroscedastic population were simulated as follows.  The explanatory
variables $x$ were simulated as standard independent multivariate
Gaussian values of dimension $d$.  The conditional mean of the
simulated population was either $x_1$ or $x_1^2$ (to consider the
effect of curvature in the population conditional quantile functions),
and the conditional variance was $\sigma^2(1 + x_2^2)$.

We generated $m=10$ datasets from the population described above with
sample size $n=1500$, considering dimensions $d=2$ and $d=5$.  We then
used QNN to estimate the conditional quantiles at $p=0.5, 0.75$, and
$0.9$ for each simulated dataset.  The estimation errors were then
used as the dependent variables in a least squares regression, with
the following mean structure

$$
\hat{Q}_y(p; x_i) - Q_y(p; x_i)$ \tilde g_p(x_i) + g_p(x_i)^2 + g_p(x_i)^3.
$$

This mean structure was fit using ordinary least squares to a dataset
consisting of all simulated values, pooling over the three values for
$p$ and over the $m$ replicates (so the overall sample size for the
least squares fit was $3mn$ since we are considering three values for
$p$).  This regression approach was adopted since it requires fewer
simulation runs than a direct approach that estimates the bias as the
sample mean of $\hat{Q} - Q$ over Monte Carlo replicates. In addition,
it gives us more insight into the structure of the bias as it relates
to factors of interest.

We noted that in the fitted regressions, the intercepts were very
small, suggesting that when $p=0.5$ there is minimal bias.  The bias
generally took the form of attenuation toward the conditional median
-- that is, when $g_p(x) > 0$ the bias was negative and with $g_p(x) <
0$ the bias was positive.  To summarize the bias, we considered a
relative bias based on the OLS fits, defined as $|{\rm bias}(p, x) /
g_p(x)|$.  For simplicity we focus on results for $p=0.75$ and take
the median of the fitted relative bias values over the $n$
observations.  These values were found to increase with $\sigma$, with
$d$, and when the true quantile functions have curvature.  However in
the situations that we considered, the relative bias tended to be less
than 10\% when $d=2$ and less than 20\% when $d=5$.
=#

function simstudy(
    x::AbstractMatrix,
    sigma::Float64,
    p::Float64,
    square::Bool;
    nrep::Int = 10,
    la::Float64 = 0.1,
)
    # The conditional mean and standard deviation
    mu = square ? x[:, 1] .^ 2 : x[:, 1]
    sd = sigma * sqrt.(1 .+ x[:, 2] .^ 2)

    # The true quantiles
    tq = [quantile(Normal(m, s), p) for (m, s) in zip(mu, sd)]

    # The estimated quantiles
    n = size(x, 1)
    qhat = zeros(n, nrep)
    for j = 1:nrep
        y = mu + sd .* randn(n)
        qr = qregnn(y, x)
        fit!(qr, p)
        qhat[:, j] = fittedvalues(qr)
    end

    # tquant are the true quantiles, qhat are the estimated quantiles
    xx = DataFrame(:tquant => repeat(tq, nrep), :qhat => vec(qhat))
    xx[:, :pc] .= p - 0.5
    return xx
end

function run_simstudy()

    # Sample size
    n = 1500

    # Target quantile at which to display results
    tq = 0.75

    for d in [2, 5]
        for square in [false, true]
            for sigma in [1.0, 2]

                # Use a fixed set of explanatory variables
                x = randn(n, d)

                # Get the estimated quantiles
                xa = []
                for p in [0.5, 0.75, 0.9]
                    xx = simstudy(x, sigma, p, square; nrep = 5, la = 0.1)
                    push!(xa, xx)
                end

                # Create a column containing the true conditional medians
                q50 = xa[1][:, :tquant]
                for j in eachindex(xa)
                    xa[j][:, :median] = q50
                end

                # tdiff is the difference between the true quantile of
                # interest and the true median.  bias is the estimation
                # error and its expected value is the bias
                xa = vcat(xa...)
                xa[:, :tdiff] = xa[:, :tquant] - xa[:, :median]
                xa[:, :bias] = xa[:, :qhat] - xa[:, :tquant]

                # If we change this formula we must change the fitted
                # relative bias values below.
                md = lm(@formula(bias ~ tdiff + tdiff^2 + tdiff^3), xa)
                println("d=$(d) sigma=$(sigma) square=$(square)")

                # Mainly we are interested in whether c[1] is small,
                # since this indicates no bias when estimating the
                # conditional median
                c = coef(md)
                println("Coefficients:", c)

                # Report results only for the tq'th probability point
                xx = xa[xa[:, :pc].==tq-0.5, :tdiff]

                # These are the fitted relative bias values
                yy = c[2] .+ c[3] * xx + c[4] * xx .^ 2
                println("Median relative bias: ", median(abs.(yy)))
            end
        end
    end
end

run_simstudy()
