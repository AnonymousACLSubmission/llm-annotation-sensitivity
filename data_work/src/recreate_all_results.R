################################################################################
### recreate_all_results.R
###
### Recreates ALL results reported in:
###   "Reliable but Sensitive: Evaluating LLM Annotation Beyond Performance"
###
### Uses the processed data files:
###   - data_work/processed/df_long.csv
###   - data_work/processed/df_agg.csv
###   - data_work/processed/prevalences.csv
###   - data_work/processed/kern_full.csv
###
### Required packages: tidyverse, lme4, lmerTest
###
### To run: Rscript data_work/src/recreate_all_results.R
### Expected runtime: ~15-30 minutes (mixed-effects models on 755k rows)
################################################################################

library(tidyverse)
library(lme4)
library(lmerTest)

### Setup paths
if (!exists("BASEDIR")) {
  BASEDIR <- "/Users/sseckman/Downloads/Annotation_Sensitivity_in_Large_Language_Models__ACL_ (1)"
}
DATA <- file.path(BASEDIR, "data_work", "processed")

cat("================================================================================\n")
cat("RECREATING ALL PAPER RESULTS\n")
cat("================================================================================\n\n")

################################################################################
### Load Data
################################################################################

cat("Loading data...\n")
df_long <- read_csv(file.path(DATA, "df_long.csv"), show_col_types = FALSE)
df_agg  <- read_csv(file.path(DATA, "df_agg.csv"), show_col_types = FALSE)
prevalences <- read_csv(file.path(DATA, "prevalences.csv"), show_col_types = FALSE)
kern_full <- read_csv(file.path(DATA, "kern_full.csv"), show_col_types = FALSE)

# The seven canonical models
MODEL_ORDER <- c("GPT-4o-mini", "GPT-5.4", "Mistral-Large-3", "Mistral-Medium-3.5",
                 "Llama-3.1-8B", "Llama-3.1-70B", "Llama-4")

# Filter to only the 7 canonical models.
# GPT-4o-mini_run2 is a second data collection run of GPT-4o-mini, used only
# for reliability checks in the Python scripts. The Python analysis_utils.py
# filters to MODEL_ORDER which excludes it (and GPT-5.4-mini).
df_long <- df_long %>% filter(model %in% MODEL_ORDER)
df_agg  <- df_agg  %>% filter(model %in% MODEL_ORDER)
prevalences <- prevalences %>% filter(model %in% MODEL_ORDER)

cat(sprintf("  df_long: %s rows\n", format(nrow(df_long), big.mark = ",")))
cat(sprintf("  Models: %s\n", paste(MODEL_ORDER, collapse = ", ")))
cat(sprintf("  Conditions: %d\n", n_distinct(df_long$condition)))
cat(sprintf("  Tweets: %d\n\n", n_distinct(df_long$tweet_id)))

################################################################################
### Eligibility Filter (Bug 1 Fix)
###
### C.HS conditions only elicit HS; the OL=0 recorded there is structural.
### C.OL conditions only elicit OL; the HS=0 recorded there is structural.
### For any cross-design analysis, filter to eligible conditions per outcome.
################################################################################

eligible_OL <- function(df) df %>% filter(task_structure != "C.HS")
eligible_HS <- function(df) df %>% filter(task_structure != "C.OL")

# 12 eligible conditions per outcome (16 total minus 4 from the excluded task_structure)
cat("Eligible conditions per outcome: 12\n")
cat(sprintf("  OL eligible rows: %s\n", format(nrow(eligible_OL(df_long)), big.mark = ",")))
cat(sprintf("  HS eligible rows: %s\n\n", format(nrow(eligible_HS(df_long)), big.mark = ",")))

################################################################################
### SECTION 4: RELIABILITY (Fleiss' Kappa within condition)
################################################################################

cat("================================================================================\n")
cat("SECTION 4: RELIABILITY\n")
cat("================================================================================\n\n")

fleiss_kappa_binary <- function(n_pos, n_raters = 3) {
  n_pos <- as.numeric(n_pos)
  n <- n_raters
  p_bar <- mean(n_pos / n)
  P_i <- (n_pos * (n_pos - 1) + (n - n_pos) * (n - n_pos - 1)) / (n * (n - 1))
  P_bar <- mean(P_i)
  P_e <- p_bar^2 + (1 - p_bar)^2
  if ((1 - P_e) == 0) return(NA_real_)
  (P_bar - P_e) / (1 - P_e)
}

# Compute Fleiss' kappa for each model × condition combination
# Using all 3 runs (R1, R2, R3) as raters within each condition
fleiss_results <- df_long %>%
  filter(model %in% MODEL_ORDER) %>%
  group_by(model, condition, task_structure, tweet_id) %>%
  summarise(
    n_pos_OL = sum(OL, na.rm = TRUE),
    n_pos_HS = sum(HS, na.rm = TRUE),
    n_raters = n(),
    .groups = "drop"
  ) %>%
  filter(n_raters == 3) %>%
  group_by(model, condition, task_structure) %>%
  summarise(
    fleiss_OL = fleiss_kappa_binary(n_pos_OL, 3),
    fleiss_HS = fleiss_kappa_binary(n_pos_HS, 3),
    .groups = "drop"
  )

# Filter to eligible outcomes
fleiss_OL <- fleiss_results %>%
  filter(task_structure != "C.HS") %>%
  pull(fleiss_OL)

fleiss_HS <- fleiss_results %>%
  filter(task_structure != "C.OL") %>%
  pull(fleiss_HS)

cat("Fleiss' kappa within-condition (3 runs):\n")
cat(sprintf("  OL range: %.2f -- %.2f\n", min(fleiss_OL, na.rm = TRUE), max(fleiss_OL, na.rm = TRUE)))
cat(sprintf("  HS range: %.2f -- %.2f\n", min(fleiss_HS, na.rm = TRUE), max(fleiss_HS, na.rm = TRUE)))
cat("  Paper reports: .84 -- .97\n\n")

# ICC per model (tweet-level)
cat("Tweet-level ICC per model (from per-model LPMs):\n")
icc_results <- tibble()
for (m in MODEL_ORDER) {
  df_m_OL <- eligible_OL(df_long) %>% filter(model == m)
  df_m_HS <- eligible_HS(df_long) %>% filter(model == m)

  fit_OL <- lmer(OL ~ 1 + (1 | tweet_id), data = df_m_OL, REML = FALSE)
  fit_HS <- lmer(HS ~ 1 + (1 | tweet_id), data = df_m_HS, REML = FALSE)

  vc_OL <- as.data.frame(VarCorr(fit_OL))
  vc_HS <- as.data.frame(VarCorr(fit_HS))
  icc_OL <- vc_OL$vcov[1] / (vc_OL$vcov[1] + sigma(fit_OL)^2)
  icc_HS <- vc_HS$vcov[1] / (vc_HS$vcov[1] + sigma(fit_HS)^2)

  icc_results <- bind_rows(icc_results, tibble(model = m, ICC_OL = icc_OL, ICC_HS = icc_HS))
  cat(sprintf("  %s: OL ICC = %.3f, HS ICC = %.3f\n", m, icc_OL, icc_HS))
}
cat("  Paper reports: Llama 3.1 8B ICC .27 OL, .33 HS; others .69--.86\n\n")

################################################################################
### SECTION 5: SENSITIVITY
################################################################################

cat("================================================================================\n")
cat("SECTION 5: SENSITIVITY\n")
cat("================================================================================\n\n")

### 5.1 Cohen's kappa between conditions (modal labels)
cat("--- 5.1 Agreement Across Defensible Designs ---\n\n")

cohens_kappa <- function(a, b) {
  valid <- !is.na(a) & !is.na(b)
  a <- a[valid]; b <- b[valid]
  if (length(a) == 0) return(NA_real_)
  po <- mean(a == b)
  pe <- mean(a) * mean(b) + (1 - mean(a)) * (1 - mean(b))
  if ((1 - pe) == 0) return(NA_real_)
  (po - pe) / (1 - pe)
}

# Get conditions that are eligible for OL
eligible_conditions_OL <- df_agg %>%
  filter(task_structure != "C.HS") %>%
  distinct(condition) %>%
  pull()

eligible_conditions_HS <- df_agg %>%
  filter(task_structure != "C.OL") %>%
  distinct(condition) %>%
  pull()

# Compute pairwise Cohen's kappa between conditions within each model
kappa_between <- tibble()
for (m in MODEL_ORDER) {
  # OL
  agg_m <- df_agg %>%
    filter(model == m, condition %in% eligible_conditions_OL) %>%
    select(tweet_id, condition, modal_OL) %>%
    pivot_wider(names_from = condition, values_from = modal_OL)

  conds <- setdiff(names(agg_m), "tweet_id")
  for (i in 1:(length(conds) - 1)) {
    for (j in (i + 1):length(conds)) {
      k <- cohens_kappa(agg_m[[conds[i]]], agg_m[[conds[j]]])
      kappa_between <- bind_rows(kappa_between,
        tibble(model = m, outcome = "OL", cond1 = conds[i], cond2 = conds[j], kappa = k))
    }
  }

  # HS
  agg_m <- df_agg %>%
    filter(model == m, condition %in% eligible_conditions_HS) %>%
    select(tweet_id, condition, modal_HS) %>%
    pivot_wider(names_from = condition, values_from = modal_HS)

  conds <- setdiff(names(agg_m), "tweet_id")
  for (i in 1:(length(conds) - 1)) {
    for (j in (i + 1):length(conds)) {
      k <- cohens_kappa(agg_m[[conds[i]]], agg_m[[conds[j]]])
      kappa_between <- bind_rows(kappa_between,
        tibble(model = m, outcome = "HS", cond1 = conds[i], cond2 = conds[j], kappa = k))
    }
  }
}

cat("Cohen's kappa between conditions (modal labels), within model:\n")
cat(sprintf("  OL range: %.2f -- %.2f\n",
            min(kappa_between$kappa[kappa_between$outcome == "OL"], na.rm = TRUE),
            max(kappa_between$kappa[kappa_between$outcome == "OL"], na.rm = TRUE)))
cat(sprintf("  HS range: %.2f -- %.2f\n",
            min(kappa_between$kappa[kappa_between$outcome == "HS"], na.rm = TRUE),
            max(kappa_between$kappa[kappa_between$outcome == "HS"], na.rm = TRUE)))
cat("  Paper reports: .57 -- .94\n\n")

### 5.1b Agreement with human reference
cat("--- Agreement with Human Reference ---\n\n")

# Human majority labels
human_maj <- kern_full %>%
  group_by(tweet_id) %>%
  summarise(
    human_OL = as.integer(mean(offensive_language, na.rm = TRUE) >= 0.5),
    human_HS = as.integer(mean(hate_speech, na.rm = TRUE) >= 0.5),
    .groups = "drop"
  )

# Cohen's kappa: each model × condition modal label vs human majority
kappa_human <- df_agg %>%
  filter(model %in% MODEL_ORDER) %>%
  inner_join(human_maj, by = "tweet_id") %>%
  group_by(model, condition, task_structure) %>%
  summarise(
    kappa_OL = cohens_kappa(modal_OL, human_OL),
    kappa_HS = cohens_kappa(modal_HS, human_HS),
    .groups = "drop"
  )

kappa_human_OL <- kappa_human %>% filter(task_structure != "C.HS") %>% pull(kappa_OL)
kappa_human_HS <- kappa_human %>% filter(task_structure != "C.OL") %>% pull(kappa_HS)

cat("Cohen's kappa vs. human majority label:\n")
cat(sprintf("  OL range: %.2f -- %.2f\n", min(kappa_human_OL, na.rm = TRUE), max(kappa_human_OL, na.rm = TRUE)))
cat(sprintf("  HS range: %.2f -- %.2f\n", min(kappa_human_HS, na.rm = TRUE), max(kappa_human_HS, na.rm = TRUE)))
cat("  Paper reports: .41 -- .62\n\n")

### 5.2 HS prevalence range across models
cat("--- HS Prevalence Range Across Models ---\n\n")

# Mean HS prevalence per model across all eligible conditions
model_hs_prev <- eligible_HS(df_long) %>%
  group_by(model) %>%
  summarise(HS_prev = mean(HS, na.rm = TRUE), .groups = "drop")

cat("HS prevalence by model:\n")
for (i in 1:nrow(model_hs_prev)) {
  cat(sprintf("  %s: %.1f%%\n", model_hs_prev$model[i], model_hs_prev$HS_prev[i] * 100))
}
cat(sprintf("  Range: %.0f%% -- %.0f%%\n", min(model_hs_prev$HS_prev) * 100, max(model_hs_prev$HS_prev) * 100))
cat("  Paper reports: 17% -- 43%\n\n")

################################################################################
### SECTION 5.2: POOLED MIXED-EFFECTS LPMs (Table 5 / tab:taskeffects_all)
################################################################################

cat("================================================================================\n")
cat("POOLED MIXED-EFFECTS LPMs (Table: taskeffects_all)\n")
cat("================================================================================\n\n")

# Prepare design factors for the pooled model
# Reference: Separate labeling (C.OL or C.HS), individual, no confidence, GPT-4o-mini
df_model_OL <- eligible_OL(df_long) %>%
  mutate(
    joint_OL_first = as.integer(task_structure == "A"),
    joint_HS_first = as.integer(task_structure == "B"),
    model = factor(model, levels = MODEL_ORDER)
  )

df_model_HS <- eligible_HS(df_long) %>%
  mutate(
    joint_OL_first = as.integer(task_structure == "A"),
    joint_HS_first = as.integer(task_structure == "B"),
    model = factor(model, levels = MODEL_ORDER)
  )

cat("--- Specification 1: Task Only ---\n")
fit_OL_task <- lmer(OL ~ joint_OL_first + joint_HS_first + confidence + batched +
                      (1 | tweet_id), data = df_model_OL, REML = FALSE)
fit_HS_task <- lmer(HS ~ joint_OL_first + joint_HS_first + confidence + batched +
                      (1 | tweet_id), data = df_model_HS, REML = FALSE)

cat("OL Task model:\n")
print(summary(fit_OL_task)$coefficients)
cat("\nHS Task model:\n")
print(summary(fit_HS_task)$coefficients)

# Extract ICC and marginal R²
vc_OL <- as.data.frame(VarCorr(fit_OL_task))
icc_OL_task <- vc_OL$vcov[1] / (vc_OL$vcov[1] + sigma(fit_OL_task)^2)
vc_HS <- as.data.frame(VarCorr(fit_HS_task))
icc_HS_task <- vc_HS$vcov[1] / (vc_HS$vcov[1] + sigma(fit_HS_task)^2)

cat(sprintf("\nOL Task ICC: %.3f (paper: 0.591)\n", icc_OL_task))
cat(sprintf("HS Task ICC: %.3f (paper: 0.563)\n\n", icc_HS_task))

# Marginal R² (Nakagawa & Schielzeth)
compute_r2 <- function(fit) {
  vc <- as.data.frame(VarCorr(fit))
  tau <- vc$vcov[1]
  sigma2 <- sigma(fit)^2
  fe_var <- var(predict(fit, re.form = NA))
  marg_r2 <- fe_var / (fe_var + tau + sigma2)
  cond_r2 <- (fe_var + tau) / (fe_var + tau + sigma2)
  list(marginal = marg_r2, conditional = cond_r2)
}

r2_OL_task <- compute_r2(fit_OL_task)
r2_HS_task <- compute_r2(fit_HS_task)
cat(sprintf("OL Task marginal R²: %.3f (paper: 0.007)\n", r2_OL_task$marginal))
cat(sprintf("HS Task marginal R²: %.3f (paper: 0.001)\n", r2_HS_task$marginal))
cat(sprintf("OL Task conditional R²: %.3f (paper: 0.594)\n", r2_OL_task$conditional))
cat(sprintf("HS Task conditional R²: %.3f (paper: 0.564)\n\n", r2_HS_task$conditional))

cat("--- Specification 2: Models (adds model fixed effects) ---\n")
fit_OL_models <- lmer(OL ~ joint_OL_first + joint_HS_first + confidence + batched +
                        model + (1 | tweet_id), data = df_model_OL, REML = FALSE)
fit_HS_models <- lmer(HS ~ joint_OL_first + joint_HS_first + confidence + batched +
                        model + (1 | tweet_id), data = df_model_HS, REML = FALSE)

cat("OL Models:\n")
print(summary(fit_OL_models)$coefficients)
cat("\nHS Models:\n")
print(summary(fit_HS_models)$coefficients)

r2_OL_models <- compute_r2(fit_OL_models)
r2_HS_models <- compute_r2(fit_HS_models)
cat(sprintf("\nOL Models marginal R²: %.3f (paper: 0.013)\n", r2_OL_models$marginal))
cat(sprintf("HS Models marginal R²: %.3f (paper: 0.021)\n\n", r2_HS_models$marginal))

cat("--- Specification 3: Random Slopes (design effects vary by model) ---\n")
fit_OL_rs <- lmer(OL ~ joint_OL_first + joint_HS_first + confidence + batched +
                    model +
                    (1 | tweet_id) +
                    (0 + joint_OL_first | model) +
                    (0 + joint_HS_first | model) +
                    (0 + confidence | model) +
                    (0 + batched | model),
                  data = df_model_OL, REML = FALSE,
                  control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 50000)))

fit_HS_rs <- lmer(HS ~ joint_OL_first + joint_HS_first + confidence + batched +
                    model +
                    (1 | tweet_id) +
                    (0 + joint_OL_first | model) +
                    (0 + joint_HS_first | model) +
                    (0 + confidence | model) +
                    (0 + batched | model),
                  data = df_model_HS, REML = FALSE,
                  control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 50000)))

cat("OL Random Slopes:\n")
print(summary(fit_OL_rs)$coefficients)
cat("\nOL variance components (random slopes):\n")
print(VarCorr(fit_OL_rs))

cat("\nHS Random Slopes:\n")
print(summary(fit_HS_rs)$coefficients)
cat("\nHS variance components (random slopes):\n")
print(VarCorr(fit_HS_rs))

# LRT: Random Slopes vs Models
lrt_OL <- anova(fit_OL_models, fit_OL_rs)
lrt_HS <- anova(fit_HS_models, fit_HS_rs)
cat(sprintf("\nLRT OL (Random Slopes vs Models): chi² = %.2f, p = %s\n",
            lrt_OL$Chisq[2], format.pval(lrt_OL$`Pr(>Chisq)`[2])))
cat(sprintf("  Paper: chi²_4 = 13,555, p < .001\n"))
cat(sprintf("LRT HS (Random Slopes vs Models): chi² = %.2f, p = %s\n",
            lrt_HS$Chisq[2], format.pval(lrt_HS$`Pr(>Chisq)`[2])))
cat(sprintf("  Paper: chi²_4 = 17,853, p < .001\n\n"))

r2_OL_rs <- compute_r2(fit_OL_rs)
r2_HS_rs <- compute_r2(fit_HS_rs)
cat(sprintf("OL Random Slopes marginal R²: %.3f (paper: 0.016)\n", r2_OL_rs$marginal))
cat(sprintf("HS Random Slopes marginal R²: %.3f (paper: 0.031)\n\n", r2_HS_rs$marginal))

################################################################################
### LOGISTIC REGRESSION (Table: task_structure_glmer)
################################################################################

cat("================================================================================\n")
cat("LOGISTIC MIXED-EFFECTS MODELS (Table: task_structure_glmer)\n")
cat("================================================================================\n\n")

cat("--- OL Task (logistic) ---\n")
fit_OL_glm_task <- glmer(OL ~ joint_OL_first + joint_HS_first + confidence + batched +
                           (1 | tweet_id),
                         data = df_model_OL, family = binomial,
                         control = glmerControl(optimizer = "bobyqa",
                                               optCtrl = list(maxfun = 100000)))
cat("OL Task (logistic):\n")
print(summary(fit_OL_glm_task)$coefficients)

cat("\n--- OL Models (logistic) ---\n")
fit_OL_glm_models <- glmer(OL ~ joint_OL_first + joint_HS_first + confidence + batched +
                             model + (1 | tweet_id),
                           data = df_model_OL, family = binomial,
                           control = glmerControl(optimizer = "bobyqa",
                                                  optCtrl = list(maxfun = 100000)))
cat("OL Models (logistic):\n")
print(summary(fit_OL_glm_models)$coefficients)

cat("\n--- HS Task (logistic) ---\n")
fit_HS_glm_task <- glmer(HS ~ joint_OL_first + joint_HS_first + confidence + batched +
                           (1 | tweet_id),
                         data = df_model_HS, family = binomial,
                         control = glmerControl(optimizer = "bobyqa",
                                               optCtrl = list(maxfun = 100000)))
cat("HS Task (logistic):\n")
print(summary(fit_HS_glm_task)$coefficients)

cat("\n--- HS Models (logistic) ---\n")
fit_HS_glm_models <- glmer(HS ~ joint_OL_first + joint_HS_first + confidence + batched +
                             model + (1 | tweet_id),
                           data = df_model_HS, family = binomial,
                           control = glmerControl(optimizer = "bobyqa",
                                                  optCtrl = list(maxfun = 100000)))
cat("HS Models (logistic):\n")
print(summary(fit_HS_glm_models)$coefficients)

################################################################################
### PER-MODEL LPMs (Table: permodel / Figure: design_by_model)
################################################################################

cat("\n================================================================================\n")
cat("PER-MODEL DESIGN EFFECTS (Table: permodel)\n")
cat("================================================================================\n\n")

permodel_results <- tibble()

for (m in MODEL_ORDER) {
  df_m_OL <- eligible_OL(df_long) %>%
    filter(model == m) %>%
    mutate(
      joint_OL_first = as.integer(task_structure == "A"),
      joint_HS_first = as.integer(task_structure == "B")
    )

  df_m_HS <- eligible_HS(df_long) %>%
    filter(model == m) %>%
    mutate(
      joint_OL_first = as.integer(task_structure == "A"),
      joint_HS_first = as.integer(task_structure == "B")
    )

  fit_m_OL <- lmer(OL ~ joint_OL_first + joint_HS_first + confidence + batched +
                     (1 | tweet_id), data = df_m_OL, REML = FALSE)
  fit_m_HS <- lmer(HS ~ joint_OL_first + joint_HS_first + confidence + batched +
                     (1 | tweet_id), data = df_m_HS, REML = FALSE)

  coef_OL <- fixef(fit_m_OL)
  coef_HS <- fixef(fit_m_HS)

  vc_OL <- as.data.frame(VarCorr(fit_m_OL))
  vc_HS <- as.data.frame(VarCorr(fit_m_HS))
  icc_m_OL <- vc_OL$vcov[1] / (vc_OL$vcov[1] + sigma(fit_m_OL)^2)
  icc_m_HS <- vc_HS$vcov[1] / (vc_HS$vcov[1] + sigma(fit_m_HS)^2)

  permodel_results <- bind_rows(permodel_results, tibble(
    model = m,
    OL_intercept   = coef_OL["(Intercept)"],
    OL_joint_OL    = coef_OL["joint_OL_first"],
    OL_joint_HS    = coef_OL["joint_HS_first"],
    OL_confidence  = coef_OL["confidence"],
    OL_batch       = coef_OL["batched"],
    OL_ICC         = icc_m_OL,
    HS_intercept   = coef_HS["(Intercept)"],
    HS_joint_OL    = coef_HS["joint_OL_first"],
    HS_joint_HS    = coef_HS["joint_HS_first"],
    HS_confidence  = coef_HS["confidence"],
    HS_batch       = coef_HS["batched"],
    HS_ICC         = icc_m_HS
  ))
}

cat("Per-model OL design effects:\n")
permodel_results %>%
  select(model, OL_intercept, OL_joint_OL, OL_joint_HS, OL_confidence, OL_batch, OL_ICC) %>%
  mutate(across(where(is.numeric), ~ round(.x, 3))) %>%
  print(n = 7, width = Inf)

cat("\nPer-model HS design effects:\n")
permodel_results %>%
  select(model, HS_intercept, HS_joint_OL, HS_joint_HS, HS_confidence, HS_batch, HS_ICC) %>%
  mutate(across(where(is.numeric), ~ round(.x, 3))) %>%
  print(n = 7, width = Inf)

# Cross-model SDs
cat("\nCross-model SD of design effects:\n")
cat(sprintf("  OL Joint OL first: %.3f (paper: .029)\n", sd(permodel_results$OL_joint_OL)))
cat(sprintf("  OL Joint HS first: %.3f (paper: .023)\n", sd(permodel_results$OL_joint_HS)))
cat(sprintf("  OL Confidence:     %.3f (paper: .018)\n", sd(permodel_results$OL_confidence)))
cat(sprintf("  OL Batch:          %.3f (paper: .067)\n", sd(permodel_results$OL_batch)))
cat(sprintf("  HS Joint OL first: %.3f (paper: .061)\n", sd(permodel_results$HS_joint_OL)))
cat(sprintf("  HS Joint HS first: %.3f (paper: .042)\n", sd(permodel_results$HS_joint_HS)))
cat(sprintf("  HS Confidence:     %.3f (paper: .052)\n", sd(permodel_results$HS_confidence)))
cat(sprintf("  HS Batch:          %.3f (paper: .061)\n\n", sd(permodel_results$HS_batch)))

# Mean cross-model SD
mean_sd_OL <- mean(c(sd(permodel_results$OL_joint_OL), sd(permodel_results$OL_joint_HS),
                     sd(permodel_results$OL_confidence), sd(permodel_results$OL_batch)))
mean_sd_HS <- mean(c(sd(permodel_results$HS_joint_OL), sd(permodel_results$HS_joint_HS),
                     sd(permodel_results$HS_confidence), sd(permodel_results$HS_batch)))
cat(sprintf("Mean cross-model SD: OL = %.1f pp, HS = %.1f pp\n", mean_sd_OL * 100, mean_sd_HS * 100))
cat("  Paper reports: 3.4 pp (OL), 5.4 pp (HS)\n\n")

################################################################################
### DESIGN EFFECTS (deff) — Variance Decomposition
### (Section: new_results2.md §1)
################################################################################

cat("================================================================================\n")
cat("DESIGN EFFECTS (deff) — Variance Decomposition\n")
cat("================================================================================\n\n")

# Compute run-level prevalences: model × condition × run → prevalence
# Each "run" is one of the 3 responders (R1, R2, R3)
run_prevs_OL <- eligible_OL(df_long) %>%
  group_by(model, condition, responder) %>%
  summarise(prev = mean(OL, na.rm = TRUE), .groups = "drop")

run_prevs_HS <- eligible_HS(df_long) %>%
  group_by(model, condition, responder) %>%
  summarise(prev = mean(HS, na.rm = TRUE), .groups = "drop")

# Variance decomposition: three-level nested structure
# Level 1: runs within (model, condition)
# Level 2: conditions within model
# Level 3: models

decompose_variance <- function(run_prevs) {
  n_tweets <- 3000
  grand_mean <- mean(run_prevs$prev)

  # Nominal sampling SD
  nominal_se <- sqrt(grand_mean * (1 - grand_mean) / n_tweets)

  # Model-condition means (average across 3 runs)
  mc_means <- run_prevs %>%
    group_by(model, condition) %>%
    summarise(mc_prev = mean(prev), .groups = "drop")

  # Run-to-run SD (within model × condition)
  run_var <- run_prevs %>%
    group_by(model, condition) %>%
    summarise(run_sd = sd(prev), .groups = "drop") %>%
    summarise(mean_run_sd = mean(run_sd, na.rm = TRUE)) %>%
    pull()

  # Model means (average across conditions)
  model_means <- mc_means %>%
    group_by(model) %>%
    summarise(model_prev = mean(mc_prev), .groups = "drop")

  # Design SD: SD of condition means within each model, then average
  design_var <- mc_means %>%
    group_by(model) %>%
    summarise(cond_sd = sd(mc_prev), .groups = "drop") %>%
    summarise(mean_cond_sd = mean(cond_sd, na.rm = TRUE)) %>%
    pull()

  # Model SD: SD of model means
  model_sd <- sd(model_means$model_prev)

  # Total SD (all run-level observations)
  total_sd <- sd(run_prevs$prev)

  # Cumulative SDs
  sd_nominal <- nominal_se
  sd_with_run <- sqrt(nominal_se^2 + run_var^2)
  sd_with_design <- sqrt(nominal_se^2 + run_var^2 + design_var^2)
  sd_total <- sqrt(nominal_se^2 + run_var^2 + design_var^2 + model_sd^2)

  # Design effect
  deff <- sd_total^2 / nominal_se^2

  list(
    nominal_se = nominal_se,
    run_sd = run_var,
    design_sd = design_var,
    model_sd = model_sd,
    sd_nominal = sd_nominal,
    sd_with_run = sd_with_run,
    sd_with_design = sd_with_design,
    sd_total = sd_total,
    deff = deff
  )
}

deff_OL <- decompose_variance(run_prevs_OL)
deff_HS <- decompose_variance(run_prevs_HS)

cat("Variance decomposition:\n")
cat(sprintf("  %-30s  OL (pp)  HS (pp)\n", "Component"))
cat(sprintf("  %-30s  %6.2f   %6.2f\n", "Sampling only (nominal SE)", deff_OL$nominal_se * 100, deff_HS$nominal_se * 100))
cat(sprintf("  %-30s  %6.2f   %6.2f\n", "Run-to-run SD", deff_OL$run_sd * 100, deff_HS$run_sd * 100))
cat(sprintf("  %-30s  %6.2f   %6.2f\n", "Design SD", deff_OL$design_sd * 100, deff_HS$design_sd * 100))
cat(sprintf("  %-30s  %6.2f   %6.2f\n", "Model SD", deff_OL$model_sd * 100, deff_HS$model_sd * 100))
cat(sprintf("\n  %-30s  %6.2f   %6.2f\n", "Cumulative: + run", deff_OL$sd_with_run * 100, deff_HS$sd_with_run * 100))
cat(sprintf("  %-30s  %6.2f   %6.2f\n", "Cumulative: + design", deff_OL$sd_with_design * 100, deff_HS$sd_with_design * 100))
cat(sprintf("  %-30s  %6.2f   %6.2f\n", "Cumulative: + model (total)", deff_OL$sd_total * 100, deff_HS$sd_total * 100))
cat(sprintf("\n  %-30s  %6.0f   %6.0f\n", "Design effect (deff)", deff_OL$deff, deff_HS$deff))
cat("\nPaper reports:\n")
cat("  Run SD: 0.36 / 0.38 pp, Design SD: 5.61 / 5.72 pp, Model SD: 3.52 / 7.12 pp\n")
cat("  Deff: 80 (OL), 114 (HS)\n")
cat("  Nominal SE: 0.75 / 0.86 pp\n\n")

# Excluding Llama 3.1 8B
run_prevs_OL_no8b <- run_prevs_OL %>% filter(model != "Llama-3.1-8B")
run_prevs_HS_no8b <- run_prevs_HS %>% filter(model != "Llama-3.1-8B")
deff_OL_no8b <- decompose_variance(run_prevs_OL_no8b)
deff_HS_no8b <- decompose_variance(run_prevs_HS_no8b)
cat(sprintf("Excluding Llama 3.1 8B — deff: OL = %.0f, HS = %.0f\n",
            deff_OL_no8b$deff, deff_HS_no8b$deff))
cat("  Paper reports: 47 (OL), 111 (HS)\n\n")

################################################################################
### HUMAN DESIGN EFFECTS (Section 3 of new_results2.md)
################################################################################

cat("================================================================================\n")
cat("HUMAN DESIGN EFFECTS\n")
cat("================================================================================\n\n")

# Kern mapping: version → design features
kern_design <- tribble(
  ~version, ~joint, ~hs_first, ~blocked,
  "A",       1,       1,         0,
  "B",       0,       1,         0,
  "C",       0,       0,         0,
  "D",       0,       1,         1,
  "E",       0,       0,         1
)

kern_full <- kern_full %>%
  left_join(kern_design, by = "version")

# Version-level prevalences (reproduces Kern Table 1)
cat("--- Version Prevalences (Kern Table 1 reproduction) ---\n")
kern_version_prev <- kern_full %>%
  group_by(version) %>%
  summarise(
    n = n(),
    OL = mean(offensive_language, na.rm = TRUE) * 100,
    HS = mean(hate_speech, na.rm = TRUE) * 100,
    .groups = "drop"
  )
print(kern_version_prev)
cat("Paper reports: OL 51.6 / 58.8 / 58.5 / 54.4 / 59.0\n")
cat("              HS 26.8 / 29.6 / 28.2 / 33.5 / 31.8\n\n")

# Cross-version Cohen's kappa (reproduces Kern Table 2)
cat("--- Cross-version kappa (Kern Table 2 reproduction) ---\n\n")

# Majority label per tweet per version
kern_maj <- kern_full %>%
  group_by(tweet_id, version) %>%
  summarise(
    OL_maj = as.integer(mean(offensive_language, na.rm = TRUE) >= 0.5),
    HS_maj = as.integer(mean(hate_speech, na.rm = TRUE) >= 0.5),
    .groups = "drop"
  ) %>%
  pivot_wider(names_from = version, values_from = c(OL_maj, HS_maj))

versions <- c("A", "B", "C", "D", "E")
kappa_cross_OL <- matrix(NA, 5, 5, dimnames = list(versions, versions))
kappa_cross_HS <- matrix(NA, 5, 5, dimnames = list(versions, versions))

for (i in 1:4) {
  for (j in (i + 1):5) {
    vi <- versions[i]; vj <- versions[j]
    kappa_cross_OL[i, j] <- cohens_kappa(
      kern_maj[[paste0("OL_maj_", vi)]], kern_maj[[paste0("OL_maj_", vj)]])
    kappa_cross_HS[i, j] <- cohens_kappa(
      kern_maj[[paste0("HS_maj_", vi)]], kern_maj[[paste0("HS_maj_", vj)]])
  }
}

cat("OL cross-version kappa:\n")
print(round(kappa_cross_OL, 3))
cat(sprintf("  Range: %.3f -- %.3f (paper: 0.635--0.742)\n\n",
            min(kappa_cross_OL, na.rm = TRUE), max(kappa_cross_OL, na.rm = TRUE)))

cat("HS cross-version kappa:\n")
print(round(kappa_cross_HS, 3))
cat(sprintf("  Range: %.3f -- %.3f (paper: 0.488--0.597)\n\n",
            min(kappa_cross_HS, na.rm = TRUE), max(kappa_cross_HS, na.rm = TRUE)))

# Human variance components
cat("--- Human Variance Components ---\n\n")

# Replicate SD: within each version, the SD of prevalence across annotator subsets
# Each version has 3 ratings per tweet (by design of Kern et al.)
# "Replicate SD" is the SD of version prevalence computed from individual annotator draws

# Get version-level prevalence by each rating position (3 per version)
kern_rep_prev <- kern_full %>%
  group_by(version, tweet_id) %>%
  mutate(rater_num = row_number()) %>%
  ungroup() %>%
  group_by(version, rater_num) %>%
  summarise(
    OL_prev = mean(offensive_language, na.rm = TRUE),
    HS_prev = mean(hate_speech, na.rm = TRUE),
    .groups = "drop"
  )

# Replicate SD (average within-version SD across the 3 rater positions)
human_rep_sd <- kern_rep_prev %>%
  group_by(version) %>%
  summarise(
    OL_rep_sd = sd(OL_prev),
    HS_rep_sd = sd(HS_prev),
    .groups = "drop"
  ) %>%
  summarise(
    OL_rep_sd = mean(OL_rep_sd),
    HS_rep_sd = mean(HS_rep_sd)
  )

cat(sprintf("Human replicate SD: OL = %.2f pp, HS = %.2f pp\n",
            human_rep_sd$OL_rep_sd * 100, human_rep_sd$HS_rep_sd * 100))
cat("  Paper reports: 0.53 / 0.59 pp\n\n")

# Design SD: SD of version means across the 5 versions
human_version_means <- kern_version_prev %>%
  summarise(
    OL_design_sd = sd(OL / 100),
    HS_design_sd = sd(HS / 100)
  )

cat(sprintf("Human design SD: OL = %.2f pp, HS = %.2f pp\n",
            human_version_means$OL_design_sd * 100, human_version_means$HS_design_sd * 100))
cat("  Paper reports: 3.31 / 2.70 pp\n\n")

# Range across versions
cat(sprintf("Range across versions: OL = %.1f pp (%s -- %s), HS = %.1f pp (%s -- %s)\n",
            max(kern_version_prev$OL) - min(kern_version_prev$OL),
            sprintf("%.1f", min(kern_version_prev$OL)),
            sprintf("%.1f", max(kern_version_prev$OL)),
            max(kern_version_prev$HS) - min(kern_version_prev$HS),
            sprintf("%.1f", min(kern_version_prev$HS)),
            sprintf("%.1f", max(kern_version_prev$HS))))
cat("  Paper reports: OL 7.3 pp (51.6--59.0), HS 6.7 pp (26.8--33.5)\n\n")

# Annotator-panel SD (cluster bootstrap)
cat("--- Annotator-Panel SD (Cluster Bootstrap, 500 reps) ---\n")
cat("  (This may take a few minutes...)\n")

set.seed(42)
n_boot <- 500

boot_panel_sd <- function(data, outcome_col) {
  annotators <- unique(data$id)
  n_ann <- length(annotators)
  boot_prevs <- numeric(n_boot)
  for (b in 1:n_boot) {
    sampled <- sample(annotators, n_ann, replace = TRUE)
    boot_data <- data[data$id %in% sampled, ]
    boot_prevs[b] <- mean(boot_data[[outcome_col]], na.rm = TRUE)
  }
  sd(boot_prevs)
}

human_panel_OL <- boot_panel_sd(kern_full, "offensive_language")
human_panel_HS <- boot_panel_sd(kern_full, "hate_speech")
cat(sprintf("Human annotator-panel SD: OL = %.2f pp, HS = %.2f pp\n",
            human_panel_OL * 100, human_panel_HS * 100))
cat("  Paper reports: 0.96 / 1.09 pp\n\n")

# Human deff
human_nom_se_OL <- sqrt(mean(kern_full$offensive_language) * (1 - mean(kern_full$offensive_language)) / 3000)
human_nom_se_HS <- sqrt(mean(kern_full$hate_speech) * (1 - mean(kern_full$hate_speech)) / 3000)

human_total_sd_OL <- sqrt(human_nom_se_OL^2 + human_rep_sd$OL_rep_sd^2 +
                           human_panel_OL^2 + human_version_means$OL_design_sd^2)
human_total_sd_HS <- sqrt(human_nom_se_HS^2 + human_rep_sd$HS_rep_sd^2 +
                           human_panel_HS^2 + human_version_means$HS_design_sd^2)
human_deff_OL <- human_total_sd_OL^2 / human_nom_se_OL^2
human_deff_HS <- human_total_sd_HS^2 / human_nom_se_HS^2

cat(sprintf("Human nominal SE: OL = %.2f pp, HS = %.2f pp\n",
            human_nom_se_OL * 100, human_nom_se_HS * 100))
cat(sprintf("Human deff: OL = %.1f, HS = %.1f\n", human_deff_OL, human_deff_HS))
cat("  Paper reports: OL 15.5, HS 13.1\n\n")

### Human OLS with annotator-clustered SEs
cat("--- Human OLS: Design Effects with Annotator-Clustered SEs ---\n\n")

# Reference: version C (separate, OL first, not blocked)
kern_full <- kern_full %>%
  mutate(version = factor(version, levels = c("C", "A", "B", "D", "E")))

# OLS for OL
human_ols_OL <- lm(offensive_language ~ joint + hs_first + blocked, data = kern_full)
cat("Human OL OLS:\n")
print(summary(human_ols_OL)$coefficients)

# OLS for HS
human_ols_HS <- lm(hate_speech ~ joint + hs_first + blocked, data = kern_full)
cat("\nHuman HS OLS:\n")
print(summary(human_ols_HS)$coefficients)

cat("\nPaper reports (percentage points):\n")
cat("  OL: joint = -5.98, hs_first = -2.14, blocked = -2.00\n")
cat("  HS: joint = -2.94, hs_first = +1.55, blocked = +3.72\n\n")

# Omnibus F-test (version differences)
cat("--- Omnibus F-test: all versions share one prevalence ---\n")
kern_full_ftest <- kern_full %>% mutate(version_f = factor(version))
ftest_OL <- lm(offensive_language ~ version_f, data = kern_full_ftest)
ftest_HS <- lm(hate_speech ~ version_f, data = kern_full_ftest)
anova_OL <- anova(ftest_OL)
anova_HS <- anova(ftest_HS)
cat(sprintf("  OL: F = %.2f, p = %.2e\n", anova_OL$`F value`[1], anova_OL$`Pr(>F)`[1]))
cat(sprintf("  HS: F = %.2f, p = %.2e\n", anova_HS$`F value`[1], anova_HS$`Pr(>F)`[1]))
cat("  Paper reports: F = 13.31, p = 1.5e-10 (OL); F = 6.70, p = 2.6e-05 (HS)\n")
cat("  (Note: paper uses annotator-clustered SEs for F-test)\n\n")

################################################################################
### HUMAN vs LLM COMPARISON TABLE
################################################################################

cat("================================================================================\n")
cat("HUMAN vs LLM ON ONE SCALE\n")
cat("================================================================================\n\n")

# LLM overall prevalence
llm_ol_prev <- mean(eligible_OL(df_long)$OL, na.rm = TRUE)
llm_hs_prev <- mean(eligible_HS(df_long)$HS, na.rm = TRUE)
human_ol_prev <- mean(kern_full$offensive_language, na.rm = TRUE)
human_hs_prev <- mean(kern_full$hate_speech, na.rm = TRUE)

cat(sprintf("%-30s  OL humans  OL LLMs  HS humans  HS LLMs\n", ""))
cat(sprintf("%-30s  %7.1f%%  %7.1f%%  %7.1f%%  %7.1f%%\n", "Prevalence",
            human_ol_prev * 100, llm_ol_prev * 100, human_hs_prev * 100, llm_hs_prev * 100))
cat(sprintf("%-30s  %7.2f   %7.2f   %7.2f   %7.2f\n", "Replicate SD (pp)",
            human_rep_sd$OL_rep_sd * 100, deff_OL$run_sd * 100,
            human_rep_sd$HS_rep_sd * 100, deff_HS$run_sd * 100))
cat(sprintf("%-30s  %7.2f       n/a   %7.2f       n/a\n", "Annotator-panel SD (pp)",
            human_panel_OL * 100, human_panel_HS * 100))
cat(sprintf("%-30s  %7.2f   %7.2f   %7.2f   %7.2f\n", "Design SD (pp)",
            human_version_means$OL_design_sd * 100, deff_OL$design_sd * 100,
            human_version_means$HS_design_sd * 100, deff_HS$design_sd * 100))
cat(sprintf("%-30s      n/a   %7.2f       n/a   %7.2f\n", "Model SD (pp)",
            deff_OL$model_sd * 100, deff_HS$model_sd * 100))
cat(sprintf("%-30s  %7.2f   %7.2f   %7.2f   %7.2f\n", "Nominal SE at n=3000 (pp)",
            human_nom_se_OL * 100, deff_OL$nominal_se * 100,
            human_nom_se_HS * 100, deff_HS$nominal_se * 100))

# LLM deff without model choice
llm_deff_no_model_OL <- (deff_OL$nominal_se^2 + deff_OL$run_sd^2 + deff_OL$design_sd^2) / deff_OL$nominal_se^2
llm_deff_no_model_HS <- (deff_HS$nominal_se^2 + deff_HS$run_sd^2 + deff_HS$design_sd^2) / deff_HS$nominal_se^2

cat(sprintf("%-30s  %7.1f   %7.1f   %7.1f   %7.1f\n", "deff (excl. model)",
            human_deff_OL, llm_deff_no_model_OL, human_deff_HS, llm_deff_no_model_HS))
cat(sprintf("%-30s      n/a   %7.1f       n/a   %7.1f\n", "deff (incl. model)",
            deff_OL$deff, deff_HS$deff))

# Design SD / Replicate SD ratio
cat(sprintf("%-30s  %7.1fx  %7.1fx  %7.1fx  %7.1fx\n", "Design SD / Replicate SD",
            human_version_means$OL_design_sd / human_rep_sd$OL_rep_sd,
            deff_OL$design_sd / deff_OL$run_sd,
            human_version_means$HS_design_sd / human_rep_sd$HS_rep_sd,
            deff_HS$design_sd / deff_HS$run_sd))

cat("\nPaper reports:\n")
cat("  Prevalence: 56.5% / 78.8% (OL), 30.0% / 33.1% (HS)\n")
cat("  Replicate SD: 0.53 / 0.36 (OL), 0.59 / 0.38 (HS)\n")
cat("  Design SD: 3.31 / 5.61 (OL), 2.70 / 5.72 (HS)\n")
cat("  Model SD: n/a / 3.52 (OL), n/a / 7.12 (HS)\n")
cat("  deff (excl model): 15.5 / 57.7 (OL), 13.1 / 45.6 (HS)\n")
cat("  deff (incl model): n/a / 80.0 (OL), n/a / 114.3 (HS)\n")
cat("  Design/Replicate ratio: 6.3x / 15.7x (OL), 4.6x / 15.2x (HS)\n\n")

################################################################################
### CONFIDENCE CALIBRATION (Section 2 of new_results2.md)
################################################################################

cat("================================================================================\n")
cat("CONFIDENCE CALIBRATION\n")
cat("================================================================================\n\n")

# Parse confidence scores from the score column
# Scores are in format like "['90', '80']" for joint conditions
# and "['85']" for separate conditions
# The alignment depends on task_structure (Bug 2 fix):
#   A/A_conf/A_batch_conf: first score = OL, second = HS
#   B/B_conf/B_batch_conf: first score = HS, second = OL (reversed!)
#   C.OL: single score = OL
#   C.HS: single score = HS

parse_confidence <- function(score_str, task_structure, label_str) {
  if (is.na(score_str) || score_str == "NA" || score_str == "") {
    return(c(OL_conf = NA_real_, HS_conf = NA_real_))
  }

  # Remove list formatting
  cleaned <- gsub("\\[|\\]|'|\"", "", score_str)
  parts <- trimws(strsplit(cleaned, ",")[[1]])
  scores <- suppressWarnings(as.numeric(parts))
  scores <- scores[!is.na(scores)]

  if (length(scores) == 0) return(c(OL_conf = NA_real_, HS_conf = NA_real_))

  if (task_structure == "C.OL") {
    return(c(OL_conf = scores[1], HS_conf = NA_real_))
  } else if (task_structure == "C.HS") {
    return(c(OL_conf = NA_real_, HS_conf = scores[1]))
  } else if (length(scores) >= 2) {
    # For joint conditions, align score to label position
    # Parse label to determine ordering
    label_cleaned <- gsub("\\[|\\]|'|\"", "", label_str)
    label_parts <- trimws(strsplit(label_cleaned, ",")[[1]])

    # Find which position has OL/HS
    ol_pos <- which(label_parts %in% c("OL", "NO"))
    hs_pos <- which(label_parts %in% c("HS", "NH"))

    if (length(ol_pos) > 0 && length(hs_pos) > 0) {
      return(c(OL_conf = scores[ol_pos[1]], HS_conf = scores[hs_pos[1]]))
    } else {
      # Fallback: use task structure ordering
      if (task_structure == "A") {
        return(c(OL_conf = scores[1], HS_conf = scores[2]))
      } else {
        return(c(OL_conf = scores[2], HS_conf = scores[1]))
      }
    }
  } else if (length(scores) == 1) {
    # Single score in a joint condition — unusual but handle it
    return(c(OL_conf = scores[1], HS_conf = scores[1]))
  }

  c(OL_conf = NA_real_, HS_conf = NA_real_)
}

# Only process confidence conditions (confidence == 1)
cat("Parsing confidence scores...\n")
df_conf <- df_long %>%
  filter(confidence == 1, model %in% MODEL_ORDER)

# This is slow on 370k+ rows; use vectorized approach where possible
# Simple vectorized parsing for the common formats
df_conf <- df_conf %>%
  mutate(
    score_clean = gsub("\\[|\\]|'|\"", "", score),
    score_parts = strsplit(score_clean, ",\\s*")
  )

# Extract scores based on position
df_conf <- df_conf %>%
  mutate(
    n_scores = sapply(score_parts, function(x) sum(!is.na(suppressWarnings(as.numeric(trimws(x)))))),
    score1 = sapply(score_parts, function(x) {
      vals <- suppressWarnings(as.numeric(trimws(x)))
      if (length(vals) >= 1 && !is.na(vals[1])) vals[1] else NA_real_
    }),
    score2 = sapply(score_parts, function(x) {
      vals <- suppressWarnings(as.numeric(trimws(x)))
      if (length(vals) >= 2 && !is.na(vals[2])) vals[2] else NA_real_
    })
  )

# Align OL/HS confidence based on task_structure (Bug 2 fix)
# A conditions: label order is [OL_label, HS_label] → score order is [OL_score, HS_score]
# B conditions: label order is [HS_label, OL_label] → score order is [HS_score, OL_score]
# C.OL: single score is OL confidence
# C.HS: single score is HS confidence
df_conf <- df_conf %>%
  mutate(
    OL_conf = case_when(
      task_structure == "C.OL" ~ score1,
      task_structure == "C.HS" ~ NA_real_,
      task_structure == "A" ~ score1,
      task_structure == "B" ~ score2,
      TRUE ~ NA_real_
    ),
    HS_conf = case_when(
      task_structure == "C.HS" ~ score1,
      task_structure == "C.OL" ~ NA_real_,
      task_structure == "A" ~ score2,
      task_structure == "B" ~ score1,
      TRUE ~ NA_real_
    )
  )

# Total confidence labels
n_OL_conf <- sum(!is.na(df_conf$OL_conf) & df_conf$task_structure != "C.HS")
n_HS_conf <- sum(!is.na(df_conf$HS_conf) & df_conf$task_structure != "C.OL")
cat(sprintf("Confidence labels: OL = %s, HS = %s\n",
            format(n_OL_conf, big.mark = ","), format(n_HS_conf, big.mark = ",")))
cat("  Paper reports: 370,600 OL, 370,928 HS\n\n")

# Mean stated confidence
mean_OL_conf <- mean(df_conf$OL_conf[df_conf$task_structure != "C.HS"], na.rm = TRUE)
mean_HS_conf <- mean(df_conf$HS_conf[df_conf$task_structure != "C.OL"], na.rm = TRUE)
cat(sprintf("Mean stated confidence: OL = %.1f, HS = %.1f\n", mean_OL_conf, mean_HS_conf))
cat("  Paper reports: 88.4 (OL), 83.8 (HS)\n\n")

# Calibration against different referents
cat("--- Calibration against referents ---\n\n")

# 1. Another run, same design: agreement among the 3 runs within a condition
# For each (model, condition, tweet), compute proportion of runs that agree with each run
run_agreement <- df_conf %>%
  filter(task_structure != "C.HS") %>%
  group_by(model, condition, tweet_id) %>%
  summarise(
    n_runs = n(),
    agree_OL = mean(OL == round(mean(OL)), na.rm = TRUE),
    mean_conf = mean(OL_conf, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  filter(n_runs == 3)

mean_within_run_OL <- mean(run_agreement$agree_OL, na.rm = TRUE)

run_agreement_HS <- df_conf %>%
  filter(task_structure != "C.OL") %>%
  group_by(model, condition, tweet_id) %>%
  summarise(
    n_runs = n(),
    agree_HS = mean(HS == round(mean(HS)), na.rm = TRUE),
    mean_conf = mean(HS_conf, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  filter(n_runs == 3)

mean_within_run_HS <- mean(run_agreement_HS$agree_HS, na.rm = TRUE)

cat(sprintf("Another run, same design — mean agreement: OL = %.1f%%, HS = %.1f%%\n",
            mean_within_run_OL * 100, mean_within_run_HS * 100))
cat("  Paper reports: 96.0% (OL), 93.5% (HS)\n\n")

# 2. Same model, different design: for each (model, tweet), does the label match
#    the model's modal label across all conditions?
model_modal_OL <- eligible_OL(df_long) %>%
  group_by(model, tweet_id) %>%
  summarise(model_modal_OL = as.integer(mean(OL, na.rm = TRUE) >= 0.5), .groups = "drop")

model_modal_HS <- eligible_HS(df_long) %>%
  group_by(model, tweet_id) %>%
  summarise(model_modal_HS = as.integer(mean(HS, na.rm = TRUE) >= 0.5), .groups = "drop")

df_conf_agree <- df_conf %>%
  filter(task_structure != "C.HS") %>%
  left_join(model_modal_OL, by = c("model", "tweet_id")) %>%
  mutate(agrees_cross_design_OL = as.integer(OL == model_modal_OL))

mean_cross_design_OL <- mean(df_conf_agree$agrees_cross_design_OL, na.rm = TRUE)

df_conf_agree_HS <- df_conf %>%
  filter(task_structure != "C.OL") %>%
  left_join(model_modal_HS, by = c("model", "tweet_id")) %>%
  mutate(agrees_cross_design_HS = as.integer(HS == model_modal_HS))

mean_cross_design_HS <- mean(df_conf_agree_HS$agrees_cross_design_HS, na.rm = TRUE)

cat(sprintf("Same model, different design — mean agreement: OL = %.1f%%, HS = %.1f%%\n",
            mean_cross_design_OL * 100, mean_cross_design_HS * 100))
cat("  Paper reports: 90.5% (OL), 86.9% (HS)\n\n")

# 3. Agreement with human annotators
df_conf_human <- df_conf %>%
  filter(task_structure != "C.HS") %>%
  left_join(human_maj, by = "tweet_id") %>%
  mutate(agrees_human_OL = as.integer(OL == human_OL))

mean_human_OL <- mean(df_conf_human$agrees_human_OL, na.rm = TRUE)

df_conf_human_HS <- df_conf %>%
  filter(task_structure != "C.OL") %>%
  left_join(human_maj, by = "tweet_id") %>%
  mutate(agrees_human_HS = as.integer(HS == human_HS))

mean_human_HS <- mean(df_conf_human_HS$agrees_human_HS, na.rm = TRUE)

cat(sprintf("Randomly drawn human annotator — mean agreement: OL = %.1f%%, HS = %.1f%%\n",
            mean_human_OL * 100, mean_human_HS * 100))
cat("  Paper reports: 69.9% (OL), 73.6% (HS)\n\n")

# 4. Human majority label
df_conf_human_maj <- df_conf %>%
  filter(task_structure != "C.HS") %>%
  left_join(human_maj, by = "tweet_id") %>%
  mutate(agrees_human_maj_OL = as.integer(OL == human_OL))

mean_human_maj_OL <- mean(df_conf_human_maj$agrees_human_maj_OL, na.rm = TRUE)

df_conf_human_maj_HS <- df_conf %>%
  filter(task_structure != "C.OL") %>%
  left_join(human_maj, by = "tweet_id") %>%
  mutate(agrees_human_maj_HS = as.integer(HS == human_HS))

mean_human_maj_HS <- mean(df_conf_human_maj_HS$agrees_human_maj_HS, na.rm = TRUE)

cat(sprintf("Human majority label — mean agreement: OL = %.1f%%, HS = %.1f%%\n",
            mean_human_maj_OL * 100, mean_human_maj_HS * 100))
cat("  Paper reports: 75.6% (OL), 80.7% (HS)\n\n")

# ECE calculations
ece_calc <- function(conf, correct, n_bins = 10) {
  valid <- !is.na(conf) & !is.na(correct)
  conf <- conf[valid]; correct <- correct[valid]
  if (length(conf) == 0) return(NA_real_)
  edges <- seq(0, 100, length.out = n_bins + 1)
  tot <- 0
  for (i in 1:n_bins) {
    a <- edges[i]; b <- edges[i + 1]
    if (i < n_bins) {
      sel <- conf >= a & conf < b
    } else {
      sel <- conf >= a & conf <= b
    }
    if (sum(sel) == 0) next
    tot <- tot + mean(sel) * abs(mean(correct[sel]) - mean(conf[sel]) / 100)
  }
  tot
}

# ECE: confidence vs agreement with human majority
ece_human_OL <- ece_calc(df_conf_human_maj$OL_conf, df_conf_human_maj$agrees_human_maj_OL)
ece_human_HS <- ece_calc(df_conf_human_maj_HS$HS_conf, df_conf_human_maj_HS$agrees_human_maj_HS)
cat(sprintf("ECE vs human majority: OL = %.3f, HS = %.3f\n", ece_human_OL, ece_human_HS))
cat("  Paper reports: 0.160 (OL), 0.111 (HS)\n\n")

# Confidence varies across designs
cat("--- Confidence is design-dependent ---\n\n")

conf_by_design_OL <- df_conf %>%
  filter(task_structure != "C.HS") %>%
  group_by(condition) %>%
  summarise(mean_OL_conf = mean(OL_conf, na.rm = TRUE), .groups = "drop")

conf_by_design_HS <- df_conf %>%
  filter(task_structure != "C.OL") %>%
  group_by(condition) %>%
  summarise(mean_HS_conf = mean(HS_conf, na.rm = TRUE), .groups = "drop")

cat("Mean OL confidence by condition:\n")
print(conf_by_design_OL, n = 20)
cat(sprintf("\nOL confidence range across designs: %.1f points\n",
            max(conf_by_design_OL$mean_OL_conf) - min(conf_by_design_OL$mean_OL_conf)))
cat("  Paper reports: 9.4 points\n\n")

cat("Mean HS confidence by condition:\n")
print(conf_by_design_HS, n = 20)
cat(sprintf("\nHS confidence range across designs: %.1f points\n",
            max(conf_by_design_HS$mean_HS_conf) - min(conf_by_design_HS$mean_HS_conf)))
cat("  Paper reports: 6.9 points\n\n")

# Batching effect on confidence
conf_batch_OL <- df_conf %>%
  filter(task_structure != "C.HS") %>%
  group_by(batched) %>%
  summarise(mean_conf = mean(OL_conf, na.rm = TRUE), .groups = "drop")

cat("OL confidence: individual vs batch:\n")
print(conf_batch_OL)
cat(sprintf("  Drop: %.1f points\n",
            conf_batch_OL$mean_conf[conf_batch_OL$batched == 0] -
              conf_batch_OL$mean_conf[conf_batch_OL$batched == 1]))
cat("  Paper reports: 91.7 → 85.1, drop of 6.6 points\n\n")

################################################################################
### FLIP RATES (from new_results2.md)
################################################################################

cat("================================================================================\n")
cat("FLIP RATES\n")
cat("================================================================================\n\n")

# A (model, tweet) pair "flips" if not all eligible conditions produce the same modal label.
flip_OL <- df_agg %>%
  filter(model %in% MODEL_ORDER, task_structure != "C.HS") %>%
  group_by(model, tweet_id) %>%
  summarise(
    n_conds = n(),
    any_flip = as.integer(min(modal_OL) != max(modal_OL)),
    .groups = "drop"
  )

flip_HS <- df_agg %>%
  filter(model %in% MODEL_ORDER, task_structure != "C.OL") %>%
  group_by(model, tweet_id) %>%
  summarise(
    n_conds = n(),
    any_flip = as.integer(min(modal_HS) != max(modal_HS)),
    .groups = "drop"
  )

cat(sprintf("OL flip rate: %.1f%% of (model, tweet) pairs\n", mean(flip_OL$any_flip) * 100))
cat(sprintf("HS flip rate: %.1f%% of (model, tweet) pairs\n", mean(flip_HS$any_flip) * 100))
cat("  Paper reports: 28.8% (OL), 34.6% (HS)\n\n")

################################################################################
### TOTAL LABEL COUNT AND HOUSEKEEPING
################################################################################

cat("================================================================================\n")
cat("HOUSEKEEPING\n")
cat("================================================================================\n\n")

total_labels_7_models <- nrow(df_long)
cat(sprintf("Total labels (7 models): %s\n", format(total_labels_7_models, big.mark = ",")))
cat("  Paper reports: 755,999\n\n")

# Per model observation count
cat("Observations per model:\n")
df_long %>%
  count(model) %>%
  arrange(model) %>%
  print(n = 7)

cat("\n================================================================================\n")
cat("DONE — All key paper results recreated.\n")
cat("================================================================================\n")
