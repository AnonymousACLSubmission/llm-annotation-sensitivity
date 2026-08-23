################################################################################
### process_raw_data.R
### Reads raw LLM annotation CSVs from data/raw/<Model>/ and the Kern et al.
### reference data, producing a unified long-format analysis dataset.
###
### Output: data/processed/df_long.csv  (one row per tweet × condition × run)
###         data/processed/df_agg.csv   (modal labels per tweet × condition)
###         data/processed/kern_full.csv (combined Kern reference data)
################################################################################

### Packages
library(tidyverse)
library(data.table)

### Stable paths: the script may be launched from any working directory.
BASEDIR  <- "/Users/sseckman/Downloads/Annotation_Sensitivity_in_Large_Language_Models__ACL_ (1)"
DATADIR  <- file.path(BASEDIR, "data_work")
RAW      <- file.path(DATADIR, "raw")
PROCESSED <- file.path(DATADIR, "processed")
KERN     <- file.path(DATADIR, "Tweets_CK")
dir.create(PROCESSED, showWarnings = FALSE, recursive = TRUE)

################################################################################
### 1. Load Kern et al. (2023) Reference Data
################################################################################

train_df <- read_csv(file.path(KERN, "full_train_s.csv"), show_col_types = FALSE)
test_df  <- read_csv(file.path(KERN, "full_test_s.csv"), show_col_types = FALSE)

# harmonize column order (test has columns in different order)
test_df <- test_df %>% select(all_of(names(train_df)))

train_df <- train_df %>% mutate(original_split = "train")
test_df  <- test_df  %>% mutate(original_split = "test")

kern_full <- bind_rows(train_df, test_df) %>%
  rename(
    tweet_id           = `tweet.id`,
    batch_tweet        = `batch.tweet`,
    hate_speech        = `hate.speech`,
    offensive_language = `offensive.language`
  )

# Reproduce Kern summary for verification
kern_summary <- kern_full %>%
  group_by(version) %>%
  summarise(
    Annotations = n(),
    Annotators  = n_distinct(id),
    OL          = mean(offensive_language, na.rm = TRUE),
    HS          = mean(hate_speech, na.rm = TRUE),
    .groups = "drop"
  )

total_ann <- sum(kern_summary$Annotations)
overall_OL <- sum(kern_summary$OL * kern_summary$Annotations) / total_ann
overall_HS <- sum(kern_summary$HS * kern_summary$Annotations) / total_ann

cat("=== Kern et al. reference ===\n")
cat(sprintf("Total annotations: %d | OL prevalence: %.3f | HS prevalence: %.3f\n",
            total_ann, overall_OL, overall_HS))

write_csv(kern_full, file.path(PROCESSED, "kern_full.csv"))

################################################################################
### 2. Load All LLM-Labeled Data
################################################################################

# Model directory mapping: folder name -> canonical model name
model_dirs <- tribble(
  ~folder,                            ~model,
  "GPT4o_mini",                       "GPT-4o-mini",
  "GPT54_mini_run1_actually_4o_mini", "GPT-4o-mini_run2",
  "GPT5.4",                           "GPT-5.4",
  "GPT5.4_mini",                      "GPT-5.4-mini",
  "Llama3.1_8B_new",                  "Llama-3.1-8B",
  "Llama3.1_70B_new",                 "Llama-3.1-70B",
  "Llama4",                           "Llama-4",
  "MistralLarge3",                    "Mistral-Large-3",
  "MistralMedium3.5_new",            "Mistral-Medium-3.5"
)

# Condition mapping based on filename patterns:
#   confno       -> base conditions (A, B, C.OL, C.HS)
#   confyes      -> conf conditions (A_conf, B_conf, C.OL_conf, C.HS_conf)
#   confno_batch -> batch conditions (A_batch, B_batch, C.OL_batch, C.HS_batch)
#   confyes_batch / batch_async_scores -> batch_conf conditions

# Core columns we always need
core_cols <- c("batch_id", "tweet_in_batch", "tweet_id", "tweet", "condition")

read_model_files <- function(folder, model_name) {
  path <- file.path(RAW, folder)
  if (!dir.exists(path)) {
    warning(sprintf("Directory not found: %s", path))
    return(NULL)
  }

  files <- list.files(path, pattern = "\\.csv$", full.names = TRUE)
  if (length(files) == 0) return(NULL)

  all_dfs <- map(files, function(f) {
    df <- read_csv(f, show_col_types = FALSE)

    # Identify which columns hold labels
    label_cols <- intersect(names(df), c("R1_label", "R2_label", "R3_label"))
    score_cols <- intersect(names(df), c("R1_score", "R2_score", "R3_score"))

    # Keep only core + label + score columns
    keep <- intersect(names(df), c(core_cols, label_cols, score_cols))
    df <- df %>% select(all_of(keep))
    df$model <- model_name
    df$source_file <- basename(f)
    df

  })

  bind_rows(all_dfs)
}

cat("\n=== Loading LLM data ===\n")
combined_raw <- map2_dfr(model_dirs$folder, model_dirs$model, function(folder, model) {
  cat(sprintf("  Loading %s (%s)...\n", model, folder))
  read_model_files(folder, model)
})

cat(sprintf("\nTotal raw rows: %d\n", nrow(combined_raw)))
cat(sprintf("Models: %s\n", paste(unique(combined_raw$model), collapse = ", ")))
cat(sprintf("Unique tweets: %d\n", n_distinct(combined_raw$tweet_id)))

################################################################################
### 3. Parse Labels into OL/HS Binary Indicators
################################################################################

# Labels come in various formats depending on the condition:
#   - Joint conditions (A, B): "OL, NH" or "OL, HS" or "['OL', 'NH']"
#   - Separate conditions (C.OL): "OL" or "NOL" or "['OL']"
#   - Separate conditions (C.HS): "HS" or "NH" or "['HS']"
#
# Strategy: detect presence of "OL" (not preceded by N) and "HS" (not preceded by N)

parse_label <- function(label) {
  if (is.na(label)) return(c(OL = NA_integer_, HS = NA_integer_))
  label <- as.character(label)
  # Remove list formatting artifacts
  label <- str_replace_all(label, "\\[|\\]|'", "")
  OL <- as.integer(str_detect(label, "(?<!N)OL"))
  HS <- as.integer(str_detect(label, "(?<!N)HS"))
  c(OL = OL, HS = HS)
}

# Pivot to long format: one row per tweet × condition × responder
df_long <- combined_raw %>%
  pivot_longer(
    cols      = matches("^R[123]_label$"),
    names_to  = "responder_label",
    values_to = "label"
  ) %>%
  mutate(
    responder = str_extract(responder_label, "R[123]")
  ) %>%
  select(-responder_label)

# If score columns exist, also pivot those and join
if (any(str_detect(names(combined_raw), "^R[123]_score$"))) {
  scores_long <- combined_raw %>%
    select(all_of(core_cols), model, source_file, matches("^R[123]_score$")) %>%
    pivot_longer(
      cols      = matches("^R[123]_score$"),
      names_to  = "responder_score",
      values_to = "score"
    ) %>%
    mutate(responder = str_extract(responder_score, "R[123]")) %>%
    select(all_of(core_cols), model, source_file, responder, score)

  df_long <- df_long %>%
    left_join(scores_long,
              by = c(core_cols, "model", "source_file", "responder"))
}

# Parse OL and HS from labels
cat("\nParsing labels...\n")
parsed <- map_dfr(df_long$label, ~ as_tibble_row(parse_label(.x)))
df_long <- bind_cols(df_long, parsed)

cat(sprintf("Long-format rows: %d\n", nrow(df_long)))

################################################################################
### 4. Add Design Factor Columns
################################################################################

# Parse condition string into task structure, batching, and confidence factors
df_long <- df_long %>%
  mutate(
    # Task structure: A (joint, OL first), B (joint, HS first), C.OL/C.HS (separate)
    task_structure = str_extract(condition, "^[ABC](\\.(?:OL|HS))?"),

    # Batching: does condition contain "batch"?
    batched = as.integer(str_detect(condition, "batch")),

    # Confidence elicitation: does condition contain "conf"?
    confidence = as.integer(str_detect(condition, "conf")),

    # Base condition (removing _conf, _batch, _batch_conf suffixes)
    base_task = str_extract(condition, "^[ABC](\\.(?:OL|HS))?"),

    # Group type for plotting
    group_type = case_when(
      !str_detect(condition, "_")           ~ "base",
      str_detect(condition, "_batch_conf$") ~ "batch_conf",
      str_detect(condition, "_batch$")      ~ "batch",
      str_detect(condition, "_conf$")       ~ "conf",
      TRUE                                  ~ "other"
    )
  )

################################################################################
### 4b. Flag Which Outcomes Were Actually Elicited
################################################################################

# parse_label() records a 0 when a construct is absent from the response
# string. In the separate-labeling conditions only one construct is requested:
# C.OL asks for offensive language only, C.HS for hate speech only. The 0s
# recorded for the construct that was never asked are structural, not
# substantive, and must be excluded from any cross-condition comparison.
# Analyses that pool conditions should filter on elicited_OL / elicited_HS
# (or use the NA-coded OL_e / HS_e columns).

df_long <- df_long %>%
  mutate(
    elicited_OL = as.integer(task_structure != "C.HS"),
    elicited_HS = as.integer(task_structure != "C.OL"),
    OL_e        = if_else(elicited_OL == 1L, OL, NA_integer_),
    HS_e        = if_else(elicited_HS == 1L, HS, NA_integer_)
  )

################################################################################
### 5. Aggregate to Modal Labels
################################################################################

df_agg <- df_long %>%
  group_by(tweet_id, tweet, condition, model, task_structure,
           batched, confidence, base_task, group_type) %>%
  summarise(
    modal_OL  = as.integer(round(mean(OL, na.rm = TRUE))),
    modal_HS  = as.integer(round(mean(HS, na.rm = TRUE))),
    elicited_OL = first(elicited_OL),
    elicited_HS = first(elicited_HS),
    n_raters  = n(),
    agree_OL  = mean(OL == modal_OL, na.rm = TRUE),
    agree_HS  = mean(HS == modal_HS, na.rm = TRUE),
    .groups   = "drop"
  )

cat(sprintf("Aggregated rows (tweet × condition × model): %d\n", nrow(df_agg)))

################################################################################
### 6. Compute Prevalences by Model × Condition
################################################################################

prevalences <- df_long %>%
  group_by(model, condition, task_structure, batched, confidence, group_type) %>%
  summarise(
    OL_prev = mean(OL, na.rm = TRUE),
    HS_prev = mean(HS, na.rm = TRUE),
    elicited_OL = first(elicited_OL),
    elicited_HS = first(elicited_HS),
    n_labels = n(),
    .groups = "drop"
  )

cat("\n=== Prevalences by model (base conditions only) ===\n")
prevalences %>%
  filter(group_type == "base") %>%
  group_by(model) %>%
  summarise(
    OL = round(mean(OL_prev), 3),
    HS = round(mean(HS_prev), 3),
    .groups = "drop"
  ) %>%
  print()

################################################################################
### 7. Save Processed Data
################################################################################

## Fail loudly if a raw-data change breaks the experiment's expected structure.
stopifnot(
  n_distinct(df_long$tweet_id) == 3000,
  all(c("R1", "R2", "R3") %in% unique(df_long$responder)),
  all(model_dirs$model %in% unique(df_long$model)),
  all(c("A", "B", "C.OL", "C.HS") %in% unique(df_long$task_structure)),
  nrow(kern_full) == 44900
)

write_csv(df_long, file.path(PROCESSED, "df_long.csv"))
write_csv(df_agg,  file.path(PROCESSED, "df_agg.csv"))
write_csv(prevalences, file.path(PROCESSED, "prevalences.csv"))

cat("\n=== Done ===\n")
cat("Saved to processed/:\n")
cat("  df_long.csv        — one row per tweet × condition × responder × model\n")
cat("  df_agg.csv         — modal labels per tweet × condition × model\n")
cat("  prevalences.csv    — mean OL/HS prevalence per model × condition\n")
cat("  kern_full.csv      — combined Kern et al. reference data\n")
cat(sprintf("\nTotal labels: %d\n", nrow(df_long)))
cat(sprintf("Models: %d | Conditions: %d | Tweets: %d\n",
            n_distinct(df_long$model),
            n_distinct(df_long$condition),
            n_distinct(df_long$tweet_id)))
