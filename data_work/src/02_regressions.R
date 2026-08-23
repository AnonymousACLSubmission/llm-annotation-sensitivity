################################################################################
### 02_regressions.R  --  every regression number in the paper
###
### Produces three tables and one intermediate file:
###
###   tab:taskeffects_all       pooled mixed-effects LPMs, three nested specs
###                             (Task / + Models / + Random Slopes) x 2 outcomes
###   tab:task_structure_glmer  the same analysis as mixed logistic regressions,
###                             four nested specs x 2 outcomes
###   tab:permodel              the LPM re-fit inside each of the seven models
###   permodel_coefs.csv        per-model design effects, read by 07_figures.R
###
### Design factors, all indicators against the reference cell (separate calls,
### individual prompts, no confidence elicited, GPT-4o-mini):
###   joint_OL_first  both items in one call, OL asked first   (task_structure A)
###   joint_HS_first  both items in one call, HS asked first   (task_structure B)
###   confidence      the prompt also asks for a confidence score
###   batched         six tweets per call instead of one
###
### "Asked first" in the interaction specification means the outcome being
### modelled was the first item in the call: joint_OL_first when modelling OL,
### joint_HS_first when modelling HS.
###
### Every fit is memoised. Set PIPELINE_REFRESH=1 to refit from scratch.
################################################################################

BASEDIR <- "/Users/sseckman/Downloads/Annotation_Sensitivity_in_Large_Language_Models__ACL_ (1)"
source(file.path(BASEDIR, "data_work", "src", "00_utils.R"))
suppressPackageStartupMessages({
  library(lme4); library(lmerTest); library(purrr)
})

say <- make_logger("02_regressions.log")

OUTCOMES <- c("OL", "HS")
## The design factor a model is interacted with in the "Task Int." specification.
FIRST_VAR <- c(OL = "joint_OL_first", HS = "joint_HS_first")

DESIGN_TERMS <- c(joint_OL_first = "Joint: OL first (vs.\\ Separate)",
                  joint_HS_first = "Joint: HS first (vs.\\ Separate)",
                  confidence     = "With Confidence (vs.\\ Without)",
                  batched        = "Batch Prompt (vs.\\ Individual)")

### -------------------------------------------------------------- analysis data

say("Loading labels ...")
long <- load_long(cols = c("tweet_id", "condition", "model", "task_structure",
                           "responder", "OL", "HS", "batched", "confidence"))

## One analysis frame per outcome: drop the conditions that never elicited that
## outcome, then build the four design indicators.
frames <- lapply(OUTCOMES, function(oc) {
  eligible(long, oc) %>%
    mutate(joint_OL_first = as.integer(task_structure == "A"),
           joint_HS_first = as.integer(task_structure == "B"),
           confidence     = as.integer(confidence),
           batched        = as.integer(batched),
           model          = factor(model, levels = MODEL_ORDER)) %>%
    filter(!is.na(.data[[oc]]))
})
names(frames) <- OUTCOMES

for (oc in OUTCOMES)
  say("  %s: %s rows, %d conditions, %d tweets", oc,
      format(nrow(frames[[oc]]), big.mark = ","),
      dplyr::n_distinct(frames[[oc]]$condition),
      dplyr::n_distinct(frames[[oc]]$tweet_id))
say("  total analysable labels: %s",
    format(nrow(frames$OL) + nrow(frames$HS) - 0, big.mark = ","))

### ------------------------------------------------------------------- formulas

## Right-hand sides shared by the linear and the logistic specifications.
rhs_task   <- "joint_OL_first + joint_HS_first + confidence + batched"
rhs_models <- paste(rhs_task, "+ model")

lpm_forms <- function(oc) list(
  task   = reformulate(c(rhs_task,   "(1 | tweet_id)"), response = oc),
  models = reformulate(c(rhs_models, "(1 | tweet_id)"), response = oc),
  ## Uncorrelated per-model random slopes, one variance component per design
  ## factor. Seven models is few, so the slopes are deliberately left
  ## independent rather than given a 4x4 covariance matrix.
  rslope = reformulate(c(rhs_models, "(1 | tweet_id)",
                         "(0 + joint_OL_first | model)",
                         "(0 + joint_HS_first | model)",
                         "(0 + confidence | model)",
                         "(0 + batched | model)"), response = oc)
)

glm_forms <- function(oc) list(
  task    = reformulate(c(rhs_task,   "(1 | tweet_id)"), response = oc),
  models  = reformulate(c(rhs_models, "(1 | tweet_id)"), response = oc),
  taskint = reformulate(c(rhs_models, sprintf("model:%s", FIRST_VAR[[oc]]),
                          "(1 | tweet_id)"), response = oc),
  desgint = reformulate(c(rhs_models, "model:batched", "model:confidence",
                          "(1 | tweet_id)"), response = oc)
)

### --------------------------------------------------------------------- fitting

LMER_CTRL  <- lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 5e4))
GLMER_CTRL <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5))

## Pooled LPMs are fitted by REML. The quantities we report from them are the
## cross-model random-slope variances, and ML underestimates those by (k-1)/k
## with only k = 7 models -- a 7 percent bias on the variances. Information
## criteria and the likelihood-ratio test are taken from ML refits further down,
## because a REML criterion is not comparable across different fixed effects.
fit_lpm <- function(oc, spec) cached(sprintf("lpm_%s_%s", oc, spec), {
  say("  fitting LPM %s / %s ...", oc, spec)
  lmer(lpm_forms(oc)[[spec]], data = frames[[oc]], REML = TRUE,
       control = LMER_CTRL)
})

fit_glm <- function(oc, spec) cached(sprintf("glm_%s_%s", oc, spec), {
  say("  fitting logistic %s / %s ...", oc, spec)
  ## Laplace approximation (nAGQ = 1, the default). nAGQ = 0 is an order of
  ## magnitude faster but shifts the fixed effects in the third decimal.
  glmer(glm_forms(oc)[[spec]], data = frames[[oc]], family = binomial,
        control = GLMER_CTRL)
})

### -------------------------------------------------------- variance quantities

## Nakagawa & Schielzeth's R-squared for a random-intercept model. For the LPM
## the level-1 variance is the residual variance; for the logistic model it is
## the logistic distribution's variance, pi^2/3.
DIST_VAR_LOGIT <- pi^2 / 3

var_parts <- function(fit) {
  vc  <- as.data.frame(VarCorr(fit))
  tau <- sum(vc$vcov[vc$grp == "tweet_id"])
  e   <- if (isGLMM(fit)) DIST_VAR_LOGIT else sigma(fit)^2
  ## Slope variances belong to neither: they are reported separately.
  slope <- vc[vc$grp != "tweet_id" & !is.na(vc$var1), c("var1", "vcov")]
  fe    <- stats::var(predict(fit, re.form = NA))
  list(tau = tau, resid = e, fe = fe, slope = slope,
       icc = tau / (tau + e),
       r2m = fe / (fe + tau + e),
       r2c = (fe + tau) / (fe + tau + e))
}

## p-value stars. lmerTest gives Satterthwaite p-values for lmer; glmer's
## summary already carries Wald p-values.
stars <- function(p) {
  if (is.na(p)) "" else if (p < .001) "***" else if (p < .01) "**" else
    if (p < .05) "*" else ""
}
cell <- function(tab, term, digits = 3) {
  if (!term %in% rownames(tab)) return("--")
  est <- tab[term, "Estimate"]
  p   <- tab[term, ncol(tab)]
  sprintf("%s%s", fmtnum(est, digits), stars(p))
}
## Minus signs must be math mode so they are typeset as minus, not hyphen.
fmtnum <- function(x, digits = 3) {
  s <- formatC(abs(x), format = "f", digits = digits)
  if (x < 0) paste0("$-$", s) else s
}

### ================================================================= LPM table

say("\n%s\nPOOLED MIXED-EFFECTS LPMs  (tab:taskeffects_all)\n%s",
    strrep("=", 78), strrep("=", 78))

lpm_specs <- c("task", "models", "rslope")
lpm_fits  <- list()
for (oc in OUTCOMES) for (sp in lpm_specs)
  lpm_fits[[paste(oc, sp)]] <- fit_lpm(oc, sp)

lpm_cols <- c(paste("OL", lpm_specs), paste("HS", lpm_specs))
lpm_tabs <- lapply(lpm_fits, function(f) summary(f)$coefficients)
lpm_vp   <- lapply(lpm_fits, var_parts)

## Report each fit in full to the log before condensing it into the table.
for (k in lpm_cols) {
  say("\n--- %s ---", k)
  say(paste(capture.output(print(round(lpm_tabs[[k]], 5))), collapse = "\n"))
  v <- lpm_vp[[k]]
  say("tau_tweet = %.3f  resid = %.3f  ICC = %.3f  R2m = %.3f  R2c = %.3f",
      v$tau, v$resid, v$icc, v$r2m, v$r2c)
  if (nrow(v$slope)) say(paste(sprintf("  Var[%s] = %.4f", v$slope$var1,
                                       v$slope$vcov), collapse = "\n"))
}

## Information criteria from ML refits (see the note at fit_lpm). refitML()
## starts from the REML solution, so this is much cheaper than a fresh fit.
lpm_ml <- cached("lpm_ml_refits", lapply(lpm_fits, lme4::refitML))
lpm_ic <- lapply(lpm_ml, function(f) c(AIC = AIC(f), BIC = BIC(f)))
for (k in lpm_cols)
  say("%-20s AIC(ML) = %.1f  BIC(ML) = %.1f   [REML criterion AIC = %.1f]",
      k, lpm_ic[[k]]["AIC"], lpm_ic[[k]]["BIC"], AIC(lpm_fits[[k]]))

row_of <- function(label, f) sprintf("%s & %s \\\\", label,
                                     paste(vapply(lpm_cols, f, character(1)),
                                           collapse = " & "))

lines <- c(
  row_of("(Intercept)", function(k) cell(lpm_tabs[[k]], "(Intercept)")),
  "\\multicolumn{7}{l}{\\textbf{Task Structure Effects}} \\\\",
  unlist(lapply(names(DESIGN_TERMS), function(tm)
    row_of(DESIGN_TERMS[[tm]], function(k) cell(lpm_tabs[[k]], tm)))),
  "\\multicolumn{7}{l}{\\textbf{Model Fixed Effects} (vs.\\ GPT-4o mini)} \\\\",
  unlist(lapply(MODEL_ORDER[-1], function(m)
    row_of(MODEL_LABELS[[m]], function(k) cell(lpm_tabs[[k]], paste0("model", m))))),
  "\\multicolumn{7}{l}{\\textbf{Cross-Model Variance Components (Random Slopes)}} \\\\",
  unlist(lapply(names(DESIGN_TERMS), function(tm)
    ## "Joint: OL first" -> "Var: Joint OL first": the inner colon would read
    ## badly after the "Var:" prefix.
    row_of(sprintf("Var: %s", gsub(":", "", sub(" \\(vs.*", "", DESIGN_TERMS[[tm]]))),
           function(k) {
             s <- lpm_vp[[k]]$slope
             if (!nrow(s) || !tm %in% s$var1) "--"
             else formatC(s$vcov[s$var1 == tm], format = "f", digits = 4)
           }))),
  "\\multicolumn{7}{l}{\\textbf{Random Effects}} \\\\",
  row_of("$\\tau_{00}$\\textsubscript{tweet\\_id}",
         function(k) formatC(lpm_vp[[k]]$tau, format = "f", digits = 3)),
  "\\midrule",
  row_of("ICC", function(k) formatC(lpm_vp[[k]]$icc, format = "f", digits = 3)),
  row_of("Marginal $R^2$", function(k) formatC(lpm_vp[[k]]$r2m, format = "f", digits = 3)),
  row_of("Conditional $R^2$", function(k) formatC(lpm_vp[[k]]$r2c, format = "f", digits = 3)),
  row_of("AIC", function(k) formatC(lpm_ic[[k]]["AIC"], format = "f", digits = 1)),
  row_of("BIC", function(k) formatC(lpm_ic[[k]]["BIC"], format = "f", digits = 1))
)

## Likelihood-ratio test of the random-slope specification against the model
## fixed-effects specification: does letting design effects vary by model help?
lrt <- lapply(OUTCOMES, function(oc)
  anova(lpm_ml[[paste(oc, "models")]], lpm_ml[[paste(oc, "rslope")]]))
names(lrt) <- OUTCOMES
lines <- c(lines,
  sprintf("LRT $\\chi^2_{4}$ (vs.\\ Models) & -- & -- & %s & -- & -- & %s \\\\",
          formatC(lrt$OL$Chisq[2], format = "f", digits = 2),
          formatC(lrt$HS$Chisq[2], format = "f", digits = 2)),
  sprintf("LRT $p$ & -- & -- & %s & -- & -- & %s \\\\",
          ifelse(lrt$OL$`Pr(>Chisq)`[2] < .001, "$<$.001",
                 formatC(lrt$OL$`Pr(>Chisq)`[2], format = "f", digits = 3)),
          ifelse(lrt$HS$`Pr(>Chisq)`[2] < .001, "$<$.001",
                 formatC(lrt$HS$`Pr(>Chisq)`[2], format = "f", digits = 3))))

write_tabular(lines, "table_taskeffects_all.tex", "lcccccc",
  header = paste("Term & OL Task & OL Models & OL Random Slopes &",
                 "HS Task & HS Models & HS Random Slopes \\\\"))

say("\nLRT random slopes vs model fixed effects:")
for (oc in OUTCOMES)
  say("  %s: chi2_%d = %s, p = %.3g", oc, lrt[[oc]]$Df[2],
      format(round(lrt[[oc]]$Chisq[2], 2), big.mark = ","),
      lrt[[oc]]$`Pr(>Chisq)`[2])

### ============================================================ logistic table

say("\n%s\nPOOLED MIXED LOGISTIC REGRESSIONS  (tab:task_structure_glmer)\n%s",
    strrep("=", 78), strrep("=", 78))

glm_specs <- c("task", "models", "taskint", "desgint")
glm_fits  <- list()
for (oc in OUTCOMES) for (sp in glm_specs)
  glm_fits[[paste(oc, sp)]] <- fit_glm(oc, sp)

glm_cols <- c(paste("OL", glm_specs), paste("HS", glm_specs))
glm_tabs <- lapply(glm_fits, function(f) summary(f)$coefficients)
glm_vp   <- lapply(glm_fits, var_parts)

for (k in glm_cols) {
  say("\n--- %s (logistic) ---", k)
  say(paste(capture.output(print(round(glm_tabs[[k]], 5))), collapse = "\n"))
  v <- glm_vp[[k]]
  say("tau_tweet = %.3f  ICC = %.3f  R2m = %.3f  R2c = %.3f",
      v$tau, v$icc, v$r2m, v$r2c)
}

grow <- function(label, f) sprintf("%s & %s \\\\", label,
                                   paste(vapply(glm_cols, f, character(1)),
                                         collapse = " & "))
## In the interaction specifications lme4 names the term model<M>:<var>; which
## way round depends on the formula, so try both.
int_cell <- function(k, m, var) {
  tab <- glm_tabs[[k]]
  oc  <- sub(" .*", "", k)
  v   <- if (var == "first") FIRST_VAR[[oc]] else var
  for (nm in c(sprintf("model%s:%s", m, v), sprintf("%s:model%s", v, m)))
    if (nm %in% rownames(tab)) return(cell(tab, nm))
  "--"
}

glines <- c(
  grow("(Intercept)", function(k) cell(glm_tabs[[k]], "(Intercept)")),
  "\\multicolumn{9}{l}{\\textbf{Task Structure Effects}} \\\\",
  unlist(lapply(names(DESIGN_TERMS), function(tm)
    grow(DESIGN_TERMS[[tm]], function(k) cell(glm_tabs[[k]], tm)))),
  "\\multicolumn{9}{l}{\\textbf{Model Fixed Effects} (vs.\\ GPT-4o mini)} \\\\",
  unlist(lapply(MODEL_ORDER[-1], function(m)
    grow(MODEL_LABELS[[m]], function(k) cell(glm_tabs[[k]], paste0("model", m))))),
  "\\multicolumn{9}{l}{\\textbf{Model $\\times$ Task Structure Interaction}} \\\\",
  unlist(lapply(MODEL_ORDER[-1], function(m)
    grow(sprintf("%s x Asked First", MODEL_LABELS[[m]]),
         function(k) int_cell(k, m, "first")))),
  "\\multicolumn{9}{l}{\\textbf{Model $\\times$ Batch/Confidence Interaction}} \\\\",
  unlist(lapply(MODEL_ORDER[-1], function(m)
    grow(sprintf("%s x Batch Prompt", MODEL_LABELS[[m]]),
         function(k) int_cell(k, m, "batched")))),
  unlist(lapply(MODEL_ORDER[-1], function(m)
    grow(sprintf("%s x With Confidence", MODEL_LABELS[[m]]),
         function(k) int_cell(k, m, "confidence")))),
  "\\multicolumn{9}{l}{\\textbf{Random Effects}} \\\\",
  grow("$\\tau_{00}$\\textsubscript{tweet\\_id}",
       function(k) formatC(glm_vp[[k]]$tau, format = "f", digits = 3)),
  "\\midrule",
  grow("ICC", function(k) formatC(glm_vp[[k]]$icc, format = "f", digits = 3)),
  grow("Marginal $R^2$", function(k) formatC(glm_vp[[k]]$r2m, format = "f", digits = 3)),
  grow("Conditional $R^2$", function(k) formatC(glm_vp[[k]]$r2c, format = "f", digits = 3)),
  grow("AIC", function(k) formatC(AIC(glm_fits[[k]]), format = "f", digits = 1)),
  grow("BIC", function(k) formatC(BIC(glm_fits[[k]]), format = "f", digits = 1))
)

write_tabular(glines, "table_task_structure_glmer.tex", "lcccccccc",
  header = paste("Term & OL Task & OL Models & OL Task Int. & OL Design Int. &",
                 "HS Task & HS Models & HS Task Int. & HS Design Int. \\\\"))

### ============================================================ per-model table

say("\n%s\nPER-MODEL DESIGN EFFECTS  (tab:permodel, fig:design_by_model)\n%s",
    strrep("=", 78), strrep("=", 78))

permodel <- cached("permodel_lpms", {
  map_dfr(OUTCOMES, function(oc) map_dfr(MODEL_ORDER, function(m) {
    say("  fitting %s / %s ...", oc, m)
    d <- frames[[oc]][frames[[oc]]$model == m, ]
    ## ML here, unlike the pooled fits above: these are single-model fits with a
    ## random intercept only, so there is no cross-model variance to debias and
    ## REML and ML agree to the reported precision.
    f <- lmer(reformulate(c(rhs_task, "(1 | tweet_id)"), response = oc),
              data = d, REML = FALSE, control = LMER_CTRL)
    tab <- summary(f)$coefficients
    v   <- var_parts(f)
    tibble::tibble(outcome = oc, model = m,
                   term = rownames(tab),
                   est = tab[, "Estimate"], se = tab[, "Std. Error"],
                   p = tab[, ncol(tab)],
                   icc = v$icc, n = nrow(d))
  }))
})

readr::write_csv(permodel, file.path(OUT, "permodel_coefs.csv"))
say("  wrote permodel_coefs.csv (%d rows)", nrow(permodel))

## The pooled task-structure LPM, in the same shape, so 07_figures.R can draw the
## pooled estimate alongside the per-model ones without refitting anything.
pooled <- map_dfr(OUTCOMES, function(oc) {
  tab <- summary(lpm_fits[[paste(oc, "task")]])$coefficients
  tibble::tibble(outcome = oc, model = "Pooled", term = rownames(tab),
                 est = tab[, "Estimate"], se = tab[, "Std. Error"],
                 p = tab[, ncol(tab)],
                 icc = lpm_vp[[paste(oc, "task")]]$icc,
                 n = nobs(lpm_fits[[paste(oc, "task")]]))
})
readr::write_csv(pooled, file.path(OUT, "pooled_coefs.csv"))
say("  wrote pooled_coefs.csv (%d rows)", nrow(pooled))

## Cross-model SD of each design effect: the quantity the paper's sensitivity
## claim rests on. It should track the random-slope SDs of the pooled model.
sds <- permodel %>%
  filter(term != "(Intercept)") %>%
  group_by(outcome, term) %>%
  summarise(sd = sd(est), mean = mean(est), .groups = "drop")

say("\nCross-model SD of design effects (pp), against the pooled random-slope SD:")
for (oc in OUTCOMES) {
  slope <- lpm_vp[[paste(oc, "rslope")]]$slope
  for (tm in names(DESIGN_TERMS)) {
    s  <- sds$sd[sds$outcome == oc & sds$term == tm]
    ps <- sqrt(slope$vcov[slope$var1 == tm])
    say("  %s %-16s SD = %5.2f pp   pooled slope SD = %5.2f pp   mean = %+6.2f pp",
        oc, tm, s * 100, ps * 100, sds$mean[sds$outcome == oc & sds$term == tm] * 100)
  }
  say("  %s mean cross-model SD over the four factors: %.1f pp", oc,
      mean(sds$sd[sds$outcome == oc]) * 100)
}

n_bigger <- sum(sds$sd > abs(sds$mean))
say("\nCross-model SD exceeds the pooled mean for %d of the %d design effects.",
    n_bigger, nrow(sds))

## Table body. Each row is one term, columns are the seven models plus the SD.
pm_wide <- function(oc, tm) {
  r <- permodel[permodel$outcome == oc & permodel$term == tm, ]
  vapply(MODEL_ORDER, function(m) {
    v <- r$est[r$model == m]
    s <- formatC(abs(v), format = "f", digits = 3)
    s <- sub("^0", "", s)
    if (v < 0) paste0("$-$", s) else s
  }, character(1))
}
pm_row <- function(label, oc, tm, indent = TRUE) {
  vals <- pm_wide(oc, tm)
  s    <- sd(permodel$est[permodel$outcome == oc & permodel$term == tm])
  sprintf("%s%s & %s & %s \\\\", if (indent) "\\quad " else "", label,
          paste(vals, collapse = " & "), sub("^0", "", formatC(s, format = "f", digits = 3)))
}
pm_icc <- function(oc) sprintf("ICC (tweet) & %s & \\\\",
  paste(vapply(MODEL_ORDER, function(m) {
    v <- unique(permodel$icc[permodel$outcome == oc & permodel$model == m])
    sub("^0", "", formatC(v, format = "f", digits = 3))
  }, character(1)), collapse = " & "))

plines <- c(
  "\\multicolumn{9}{l}{\\textbf{Offensive Language}} \\\\",
  pm_row("Intercept (Separate, individual, no confidence)", "OL", "(Intercept)", FALSE),
  unlist(lapply(names(DESIGN_TERMS), function(tm)
    pm_row(DESIGN_TERMS[[tm]], "OL", tm))),
  pm_icc("OL"),
  "\\midrule",
  "\\multicolumn{9}{l}{\\textbf{Hate Speech}} \\\\",
  pm_row("Intercept (Separate, individual, no confidence)", "HS", "(Intercept)", FALSE),
  unlist(lapply(names(DESIGN_TERMS), function(tm)
    pm_row(DESIGN_TERMS[[tm]], "HS", tm))),
  pm_icc("HS"),
  "\\midrule",
  sprintf("Observations & %s & \\\\",
          paste(vapply(MODEL_ORDER, function(m)
            format(unique(permodel$n[permodel$model == m & permodel$outcome == "OL"]),
                   big.mark = ","), character(1)), collapse = " & "))
)

write_tabular(plines, "table_permodel.tex", "lccccccc|c",
  header = sprintf("Term & %s & SD \\\\",
                   paste(MODEL_LABELS[MODEL_ORDER], collapse = " & ")))

say("\nPer-model ICCs (tweet level):")
for (oc in OUTCOMES) {
  ic <- permodel %>% filter(outcome == oc) %>% distinct(model, icc)
  say("  %s: %s", oc, paste(sprintf("%s %.3f", ic$model, ic$icc), collapse = "  "))
  six <- ic$icc[ic$model != "Llama-3.1-8B"]
  say("     Llama-3.1-8B %.3f, other six %.3f-%.3f",
      ic$icc[ic$model == "Llama-3.1-8B"], min(six), max(six))
}

## How often is the cheapest model the extreme estimate? The caption claims five
## of the eight design effects.
extreme <- permodel %>% filter(term != "(Intercept)") %>%
  group_by(outcome, term) %>%
  summarise(who = model[which.max(abs(est - mean(est)))], .groups = "drop")
say("\nMost extreme model per design effect:")
for (i in seq_len(nrow(extreme)))
  say("  %s %-16s %s", extreme$outcome[i], extreme$term[i], extreme$who[i])
say("Llama-3.1-8B is the extreme estimate for %d of %d design effects.",
    sum(extreme$who == "Llama-3.1-8B"), nrow(extreme))

say("\nDone.")
