################################################################################
### run_from_raw.R  --  one command from raw CSVs to every paper artifact
###
### Default: rebuild processed data, invalidate cached fits, and refit everything.
### For a fast integrity check using existing model-fit caches:
###   USE_CACHE=1 Rscript data_work/src/run_from_raw.R
################################################################################

BASEDIR <- "/Users/sseckman/Downloads/Annotation_Sensitivity_in_Large_Language_Models__ACL_ (1)"
SRC <- file.path(BASEDIR, "data_work", "src")
RSCRIPT <- file.path(R.home("bin"), "Rscript")

run_stage <- function(file, env = character()) {
  cat(sprintf("\n%s\n== %s\n%s\n", strrep("=", 78), file, strrep("=", 78)))
  status <- system2(RSCRIPT, shQuote(file.path(SRC, file)), env = env)
  if (status != 0) stop(file, " failed with exit status ", status, call. = FALSE)
}

required_raw <- c(
  file.path(BASEDIR, "data_work", "Tweets_CK", "full_train_s.csv"),
  file.path(BASEDIR, "data_work", "Tweets_CK", "full_test_s.csv")
)
if (any(!file.exists(required_raw))) {
  stop("Missing raw reference files: ", paste(required_raw[!file.exists(required_raw)], collapse = ", "))
}
raw_csv <- list.files(file.path(BASEDIR, "data_work", "raw"), pattern = "[.]csv$",
                      recursive = TRUE, full.names = TRUE)
if (!length(raw_csv)) stop("No raw LLM annotation CSVs found.")
cat(sprintf("Found %d raw LLM CSVs and both Kern reference files.\n", length(raw_csv)))

run_stage("process_raw_data.R")

processed <- file.path(BASEDIR, "data_work", "processed",
                       c("df_long.csv", "df_agg.csv", "prevalences.csv", "kern_full.csv"))
if (any(!file.exists(processed))) stop("Raw processing did not create every required dataset.")

use_cache <- identical(Sys.getenv("USE_CACHE"), "1")
env <- if (use_cache) character() else "PIPELINE_REFRESH=1"
cat(if (use_cache) "Using existing fit caches for the analysis stages.\n" else
      "Invalidating fit caches and rebuilding every analysis from processed data.\n")
run_stage("run_all.R", env)
cat("\nRaw-data-to-paper pipeline completed successfully.\n")
