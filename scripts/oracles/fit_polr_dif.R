#!/usr/bin/env Rscript
# Canonical proportional-odds DIF oracle (R MASS::polr).
#
# Fits the three nested cumulative-logit models that metajudge.dif.logistic_dif
# fits in scipy, and prints the likelihood-ratio chi-square tests, their
# p-values, and the Nagelkerke pseudo-R-squared change. This is the external
# reference the pinned constants in tests/test_dif.py are checked against
# (SPEC R19: numerical correctness is non-compressible).
#
# Input CSV columns (one row per rated observation, built by gen_olr_oracle.py
# exactly as the engine builds its design):
#   score : ordinal response (integer category)
#   cond  : the conditioner, already standardized (z-scored, ddof=0)
#   group : focal indicator (1.0 focal, 0.0 reference)
#
# Usage:  Rscript fit_polr_dif.R <design.csv>
# Requires: R with the MASS package (install.packages("MASS")).

suppressMessages(library(MASS))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) {
  stop("usage: Rscript fit_polr_dif.R <design.csv>")
}
d <- read.csv(args[[1L]])
d$score <- factor(d$score, ordered = TRUE)

# Three nested proportional-odds models: conditioner only; + group (uniform DIF);
# + group x conditioner interaction (nonuniform DIF).
m1 <- polr(score ~ cond, data = d, method = "logistic", Hess = TRUE)
m2 <- polr(score ~ cond + group, data = d, method = "logistic", Hess = TRUE)
m3 <- polr(score ~ cond * group, data = d, method = "logistic", Hess = TRUE)

ll1 <- as.numeric(logLik(m1))
ll2 <- as.numeric(logLik(m2))
ll3 <- as.numeric(logLik(m3))
n <- nrow(d)

# Null (intercept-only) log-likelihood from the observed category frequencies,
# matching metajudge.dif._nagelkerke's reference point.
tab <- table(d$score)
ll0 <- sum(tab * log(tab / n))
nagelkerke <- function(llm) {
  cox_snell <- 1 - exp(2 * (ll0 - llm) / n)
  cox_snell / (1 - exp(2 * ll0 / n))
}

chi2_total <- -2 * (ll1 - ll3)
chi2_uniform <- -2 * (ll1 - ll2)
chi2_nonuniform <- -2 * (ll2 - ll3)
r2_delta <- nagelkerke(ll3) - nagelkerke(ll1)

cat(sprintf("chi2_total=%.6f\n", chi2_total))
cat(sprintf("chi2_uniform=%.6f\n", chi2_uniform))
cat(sprintf("chi2_nonuniform=%.6f\n", chi2_nonuniform))
cat(sprintf("p_total=%.8f\n", pchisq(chi2_total, df = 2, lower.tail = FALSE)))
cat(sprintf("p_uniform=%.8f\n", pchisq(chi2_uniform, df = 1, lower.tail = FALSE)))
cat(sprintf("p_nonuniform=%.8f\n", pchisq(chi2_nonuniform, df = 1, lower.tail = FALSE)))
cat(sprintf("nagelkerke_r2_delta=%.6f\n", r2_delta))
