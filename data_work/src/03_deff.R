################################################################################
### 03_deff.R  --  crossed-REML instrument-variance table (tab:deff)
###
### Unit of analysis: one prevalence per model x task design x run,
### 7 x 12 x 3 = 252 per outcome. A crossed random-effects model jointly
### estimates model, task-design, model-by-design, and run-to-run variance.
### Task-design variance combines the design and model-by-design components,
### because choosing a design exposes an application to both. Nominal binomial
### sampling variance is then added independently. Cumulative deff is cumulative
### variance divided by nominal sampling variance.
################################################################################

BASEDIR <- "/Users/sseckman/Downloads/Annotation_Sensitivity_in_Large_Language_Models__ACL_ (1)"
source(file.path(BASEDIR, "data_work", "src", "00_utils.R"))
suppressPackageStartupMessages({ library(purrr); library(lme4) })

say <- make_logger("03_deff.log")
N_ITEMS  <- 3000
Z95      <- 1.959964
OUTCOMES <- c("OL", "HS")

say("Loading labels ...")
long <- load_long(cols = c("tweet_id", "condition", "model", "task_structure",
                           "responder", "OL", "HS"))
run_prev <- map_dfr(OUTCOMES, function(oc) {
  eligible(long, oc) %>%
    group_by(model, condition, responder) %>%
    summarise(prev = mean(.data[[oc]], na.rm = TRUE), .groups = "drop") %>%
    mutate(outcome = oc)
})
stopifnot(all(table(run_prev$outcome) == 252))
readr::write_csv(run_prev, file.path(OUT, "deff_run_prevalence.csv"))

fit_one <- function(oc, d = run_prev) {
  x <- d[d$outcome == oc, ]
  z <- crossed_prevalence_components(x, N_ITEMS)
  say("  %s: %d prevalences; %d models x %d designs x %d runs", oc, nrow(x),
      n_distinct(x$model), n_distinct(x$condition), n_distinct(x$responder))
  z
}
dec <- setNames(lapply(OUTCOMES, fit_one), OUTCOMES)

LABELS <- c(nominal = "Sampling only (nominal)",
            run = "Run-to-run variation",
            design = "Task design, incl. Model x Design",
            model = "Model choice")

say("\n%s\nCROSSED REML INSTRUMENT VARIANCE  (tab:deff)\n%s",
    strrep("=", 84), strrep("=", 84))
say("%-36s %23s %23s", "", "Offensive language", "Hate speech")
say("%-36s %7s %7s %7s %7s   %7s %7s %7s %7s", "Source", "Comp", "Cum", "+-95", "deff",
    "Comp", "Cum", "+-95", "deff")
for (k in names(LABELS)) {
  say("%-36s %7.2f %7.2f %7.1f %7.1f   %7.2f %7.2f %7.1f %7.1f", LABELS[[k]],
      dec$OL$component_sd[[k]] * 100, dec$OL$cumulative_sd[[k]] * 100,
      Z95 * dec$OL$cumulative_sd[[k]] * 100, dec$OL$cumulative_deff[[k]],
      dec$HS$component_sd[[k]] * 100, dec$HS$cumulative_sd[[k]] * 100,
      Z95 * dec$HS$cumulative_sd[[k]] * 100, dec$HS$cumulative_deff[[k]])
}
for (oc in OUTCOMES) {
  r <- dec[[oc]]$raw_var
  say("%s raw REML variance components: design %.8f; model:design %.8f; model %.8f; run %.8f",
      oc, r[["design"]], r[["model_design"]], r[["model"]], r[["run"]])
}

## Robustness: refit after excluding the least reliable model.
dec6 <- setNames(lapply(OUTCOMES, function(oc) {
  fit_one(oc, run_prev[run_prev$model != "Llama-3.1-8B", ])
}), OUTCOMES)
say("Excluding Llama 3.1 8B: final cumulative deff OL %.1f, HS %.1f",
    dec6$OL$cumulative_deff[["model"]], dec6$HS$cumulative_deff[["model"]])

## Within-model design variation. With one model fixed, a random-intercept model
## for condition separates condition variance from run-to-run residual variance.
permodel_deff <- map_dfr(OUTCOMES, function(oc) map_dfr(MODEL_ORDER, function(m) {
  d <- run_prev[run_prev$outcome == oc & run_prev$model == m, ]
  fit <- lmer(prev ~ 1 + (1 | condition), data = d, REML = TRUE,
              control = lmerControl(check.conv.singular = "ignore"))
  vc <- as.data.frame(VarCorr(fit))
  v_design <- vc$vcov[vc$grp == "condition"]
  v_run <- vc$vcov[vc$grp == "Residual"]
  p0 <- mean(d$prev); v_nominal <- p0 * (1 - p0) / N_ITEMS
  tibble::tibble(outcome = oc, model = m, prev = p0,
                 sd_nominal = sqrt(v_nominal), sd_run = sqrt(v_run),
                 sd_design = sqrt(v_design),
                 sd_total = sqrt(v_nominal + v_run + v_design),
                 deff = (v_nominal + v_run + v_design) / v_nominal)
}))
readr::write_csv(permodel_deff, file.path(OUT, "deff_per_model.csv"))

num <- function(x, d = 2) formatC(x, format = "f", digits = d)
ROW_END <- intToUtf8(c(92, 92))
lines <- vapply(names(LABELS), function(k) sprintf(
  "%s & %s & %s & $\\pm$%s & %s & %s & %s & $\\pm$%s & %s \\\\",
  LABELS[[k]],
  num(dec$OL$component_sd[[k]] * 100), num(dec$OL$cumulative_sd[[k]] * 100),
  num(Z95 * dec$OL$cumulative_sd[[k]] * 100, 1), num(dec$OL$cumulative_deff[[k]], 1),
  num(dec$HS$component_sd[[k]] * 100), num(dec$HS$cumulative_sd[[k]] * 100),
  num(Z95 * dec$HS$cumulative_sd[[k]] * 100, 1), num(dec$HS$cumulative_deff[[k]], 1)),
  character(1))
write_tabular(lines, "table_deff.tex", "@{}lrrrrrrrr@{}", header = c(
  r"( & \multicolumn{4}{c}{\textbf{Offensive language}} & \multicolumn{4}{c}{\textbf{Hate speech}} \\)",
  r"(\cmidrule(lr){2-5}\cmidrule(lr){6-9})",
  r"(Variance source added & Component SD & SD$_{\mathrm{cum}}$ & $\pm$95\% & $deff_{\mathrm{cum}}$ & Component SD & SD$_{\mathrm{cum}}$ & $\pm$95\% & $deff_{\mathrm{cum}}$ \\)"))

components <- map_dfr(OUTCOMES, function(oc) {
  z <- dec[[oc]]
  tibble::tibble(
    outcome = oc, component = names(LABELS),
    component_variance = unname(z$component_var),
    component_sd_pp = unname(z$component_sd) * 100,
    cumulative_variance = unname(z$cumulative_var),
    cumulative_sd_pp = unname(z$cumulative_sd) * 100,
    ci95_pp = Z95 * unname(z$cumulative_sd) * 100,
    cumulative_deff = unname(z$cumulative_deff),
    final_deff_six_models = dec6[[oc]]$cumulative_deff[["model"]])
})
readr::write_csv(components, file.path(OUT, "deff_components.csv"))
raw <- map_dfr(OUTCOMES, function(oc) tibble::tibble(
  outcome = oc, component = names(dec[[oc]]$raw_var),
  variance = unname(dec[[oc]]$raw_var), sd_pp = sqrt(variance) * 100))
readr::write_csv(raw, file.path(OUT, "deff_reml_components_raw.csv"))
say("Wrote table_deff.tex, deff_components.csv, and supporting exports.")
say("\nDone.")
