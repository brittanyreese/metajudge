# Fits MASS::polr(score ~ theta + group) on a CSV the Python harness writes and prints
# the two recovered coefficients, one per line as "theta <value>" / "group <value>".
# Run indirectly via scripts/oracles/gen_dgp_recovery.py.
args <- commandArgs(trailingOnly = TRUE)
ok <- requireNamespace("MASS", quietly = TRUE)
if (!ok) { quit(status = 2) }
d <- read.csv(args[1])
d$score <- factor(d$score, ordered = TRUE)
m <- MASS::polr(score ~ theta + group, data = d, Hess = TRUE)
co <- coef(m)
cat(sprintf("theta %.6f\n", co[["theta"]]))
cat(sprintf("group %.6f\n", co[["group"]]))
