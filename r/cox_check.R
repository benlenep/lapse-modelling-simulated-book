# Milestone 3: fit the same Cox model in R and save the coefficients for comparison.
# Run from the repository root, after notebook 03 has written data/members.csv:
#   Rscript r/cox_check.R

suppressPackageStartupMessages(library(survival))

members <- read.csv("data/members.csv", stringsAsFactors = FALSE)
members$age_band     <- factor(members$age_band, levels = c("18-30", "31-45", "46-60", "61+"))
members$product_tier <- factor(members$product_tier, levels = c("basic", "mid", "top"))
members$channel      <- factor(members$channel, levels = c("direct", "broker", "online"))

fit <- coxph(Surv(duration, lapsed) ~ age_band + product_tier + channel + premium_increase,
             data = members, ties = "efron")

dir.create("r_outputs", showWarnings = FALSE)
write.csv(data.frame(term = names(coef(fit)), estimate = unname(coef(fit))),
          "r_outputs/cox_coefs.csv", row.names = FALSE)

print(round(coef(fit), 4))

km <- survfit(Surv(duration, lapsed) ~ 1, data = members)
cat("\nKaplan-Meier survival at 12 and 24 months (compare with notebook 02):\n")
print(round(summary(km, times = c(12, 24))$surv, 4))
