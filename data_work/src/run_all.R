################################################################################
### run_all.R  --  every number and every figure in the paper, in order.
###
###   Rscript data_work/src/run_all.R            # reuse cached model fits
###   PIPELINE_REFRESH=1 Rscript data_work/src/run_all.R   # refit everything
###
### Each stage is a separate R process, so a failure in one cannot leave a
### half-built object behind for the next. 02 is the slow one: eight mixed
### logistic regressions over 756k rows, roughly an hour cold, seconds warm.
###
### The only figure in the paper that this pipeline does not draw is method6.png,
### a hand-made diagram of the prompt conditions.
################################################################################

BASEDIR <- "/Users/sseckman/Downloads/Annotation_Sensitivity_in_Large_Language_Models__ACL_ (1)"
SRC     <- file.path(BASEDIR, "data_work", "src")

STAGES <- c(
  "01_agreement.R"    = "reliability / sensitivity / validity ladder, ICCs, fig:agreements",
  "02_regressions.R"  = "pooled LPMs and mixed logits, per-model design effects",
  "03_deff.R"         = "crossed-REML instrument variance and cumulative deff",
  "04_human_design.R" = "human questionnaire-version effects (tab:human_llm)",
  "05_confidence.R"   = "elicited confidence, calibration, flips (fig:confidence)",
  "06_prevalence.R"   = "prevalence heatmap and run-to-run stability",
  "07_figures.R"      = "fig:design_by_model, from 02's coefficient exports",
  "08_paper_tables.R" = "framework and annotation-cost tables"
)

RSCRIPT <- file.path(R.home("bin"), "Rscript")
t_all   <- Sys.time()
failed  <- character(0)

for (f in names(STAGES)) {
  cat(sprintf("\n%s\n== %-20s %s\n%s\n", strrep("=", 78), f, STAGES[[f]],
              strrep("=", 78)))
  t0 <- Sys.time()
  st <- system2(RSCRIPT, shQuote(file.path(SRC, f)))
  dt <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  if (st != 0) {
    failed <- c(failed, f)
    cat(sprintf("\n!! %s exited %d after %.0fs\n", f, st, dt))
  } else {
    cat(sprintf("\n-- %s ok (%.0fs)\n", f, dt))
  }
}

cat(sprintf("\n%s\nTotal %.1f min. %d of %d stages ok.\n", strrep("=", 78),
            as.numeric(difftime(Sys.time(), t_all, units = "mins")),
            length(STAGES) - length(failed), length(STAGES)))
if (length(failed)) {
  cat("Failed: ", paste(failed, collapse = ", "), "\n")
  quit(status = 1)
}

### ------------------------------------------------------- what should now exist

OUT   <- file.path(BASEDIR, "data_work", "outputs")
PLOTS <- file.path(BASEDIR, "plots")

FIGURES <- c("agreements_few_new_conditions_2x1.pdf",
             "design_effects_families.pdf",
             "confidence_calibration_and_flips.pdf",
             "confidence_calibration_two_panel.pdf",
             "confidence_calibration_two_panel.png",
             "confidence_calibration_per_model.pdf",
             "prevalence_heatmap_all_models_vs_model_mean_stacked.pdf",
             "prevalence_stability_runs.pdf")

TABLES <- c("table_agreement_ladder.tex", "table_taskeffects_all.tex",
            "table_task_structure_glmer.tex", "table_permodel.tex",
            "table_deff.tex", "table_human_llm.tex",
            "table_confidence_permodel.tex", "table_framework.tex",
            "table_costs.tex")

check <- function(dir, files, what) {
  miss <- files[!file.exists(file.path(dir, files))]
  cat(sprintf("\n%s: %d of %d present\n", what, length(files) - length(miss),
              length(files)))
  for (f in setdiff(files, miss))
    cat(sprintf("  ok   %-58s %s\n", f,
                format(file.info(file.path(dir, f))$mtime, "%H:%M:%S")))
  for (f in miss) cat(sprintf("  MISSING %s\n", f))
  length(miss)
}

n <- check(PLOTS, FIGURES, "Figures") + check(OUT, TABLES, "Tables")
if (n > 0) quit(status = 1)
cat("\nAll paper artefacts rebuilt from R.\n")
