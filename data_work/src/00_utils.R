################################################################################
### 00_utils.R  --  shared helpers for the annotation-sensitivity analyses
###
### Sourced by every numbered script. Defines paths, the canonical eligibility
### filter, agreement and calibration estimators, the plotting theme, and the
### LaTeX table writer. No analysis happens here.
###
### WHY THE ELIGIBILITY FILTER EXISTS
### --------------------------------
### process_raw_data.R records an *absent* outcome as 0 rather than NA. In the
### separate-labeling conditions only one construct is elicited: C.OL asks for
### offensive language only, C.HS for hate speech only. The parser therefore
### writes HS = 0 for every tweet in C.OL and OL = 0 for every tweet in C.HS,
### because the returned string contains no HS (resp. OL) token.
###
### Any analysis pooling conditions without filtering treats those structural
### zeros as substantive negatives. For cross-design comparisons this is fatal:
### every tweet labeled OL somewhere automatically "disagrees" with the C.HS
### conditions, inflating apparent instability (88.5% against the correct
### 27.1% for GPT-4o-mini). eligible() is the single canonical filter.
################################################################################

suppressPackageStartupMessages({
  library(data.table)
  library(dplyr)
  library(tidyr)
  library(ggplot2)
})

if (!exists("BASEDIR")) {
  BASEDIR <- "/Users/sseckman/Downloads/Annotation_Sensitivity_in_Large_Language_Models__ACL_ (1)"
}
DATA  <- file.path(BASEDIR, "data_work", "processed")
OUT   <- file.path(BASEDIR, "data_work", "outputs")
PLOTS <- file.path(BASEDIR, "plots")
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)

### ---------------------------------------------------------------- constants

## The seven models reported in the paper. GPT-4o-mini is the reference
## category throughout. The processed data also contain GPT-4o-mini_run2 (a
## re-collection ten months later, used only for the temporal-drift check) and
## GPT-5.4-mini; both are excluded from every reported result.
MODEL_ORDER <- c("GPT-4o-mini", "GPT-5.4", "Mistral-Large-3", "Mistral-Medium-3.5",
                 "Llama-3.1-8B", "Llama-3.1-70B", "Llama-4")

MODEL_LABELS <- c("GPT-4o-mini" = "GPT-4o-mini", "GPT-5.4" = "GPT-5.4",
                  "Mistral-Large-3" = "Mistral Large 3",
                  "Mistral-Medium-3.5" = "Mistral Medium 3.5",
                  "Llama-3.1-8B" = "Llama 3.1 8B", "Llama-3.1-70B" = "Llama 3.1 70B",
                  "Llama-4" = "Llama 4")

MODEL_FAM <- c("GPT-4o-mini" = "OpenAI", "GPT-5.4" = "OpenAI",
               "Mistral-Large-3" = "Mistral", "Mistral-Medium-3.5" = "Mistral",
               "Llama-3.1-8B" = "Meta", "Llama-3.1-70B" = "Meta", "Llama-4" = "Meta")

## Okabe-Ito: colour encodes family, shape encodes size within family.
FAM_COLORS  <- c(OpenAI = "#0072B2", Meta = "#E69F00", Mistral = "#009E73")
## Within a family the larger model comes first and takes the circle, the next
## the square, the third the triangle. All three are fillable so a figure can
## show them hollow (fill = NA) or solid.
MODEL_SHAPE <- c("GPT-4o-mini" = 21, "GPT-5.4" = 22,
                 "Mistral-Large-3" = 21, "Mistral-Medium-3.5" = 22,
                 "Llama-3.1-8B" = 21, "Llama-3.1-70B" = 22, "Llama-4" = 24)

OL_COLOR <- "#0072B2"
HS_COLOR <- "#D55E00"

## Conditions in which each outcome was actually requested.
## task_structure: A    = joint call, OL asked first
##                 B    = joint call, HS asked first
##                 C.OL = separate call, OL only
##                 C.HS = separate call, HS only
NOT_ELICITED <- c(OL = "C.HS", HS = "C.OL")

## Human-readable condition labels, used on figure axes and in tables.
COND_LABELS <- c(
  "A" = "Joint, OL first: Base",           "A_conf" = "Joint, OL first: + Conf.",
  "A_batch" = "Joint, OL first: Batch",    "A_batch_conf" = "Joint, OL first: Batch + Conf.",
  "B" = "Joint, HS first: Base",           "B_conf" = "Joint, HS first: + Conf.",
  "B_batch" = "Joint, HS first: Batch",    "B_batch_conf" = "Joint, HS first: Batch + Conf.",
  "C.OL" = "Separate: Base",               "C.OL_conf" = "Separate: + Conf.",
  "C.OL_batch" = "Separate: Batch",        "C.OL_batch_conf" = "Separate: Batch + Conf.",
  "C.HS" = "Separate: Base",               "C.HS_conf" = "Separate: + Conf.",
  "C.HS_batch" = "Separate: Batch",        "C.HS_batch_conf" = "Separate: Batch + Conf."
)

### ------------------------------------------------------------------ loading

eligible <- function(df, outcome) {
  stopifnot(outcome %in% names(NOT_ELICITED))
  df[df$task_structure != NOT_ELICITED[[outcome]], , drop = FALSE]
}

load_long <- function(cols = NULL, models = MODEL_ORDER) {
  d <- data.table::fread(file.path(DATA, "df_long.csv"), select = cols,
                         showProgress = FALSE)
  if (!is.null(models)) d <- d[d$model %in% models, ]
  ## OL / HS are 0/1 flags. `label` and `score` are Python list literals
  ## ("['OL', 'NH']", "['90', '80']") and must stay character for 05 to parse.
  for (v in intersect(c("OL", "HS"), names(d))) {
    d[[v]] <- suppressWarnings(as.numeric(d[[v]]))
  }
  for (v in intersect(c("label", "score"), names(d))) {
    d[[v]] <- as.character(d[[v]])
  }
  as.data.frame(d)
}

load_agg <- function(models = MODEL_ORDER) {
  d <- data.table::fread(file.path(DATA, "df_agg.csv"), showProgress = FALSE)
  if (!is.null(models)) d <- d[d$model %in% models, ]
  as.data.frame(d)
}

load_kern <- function() {
  as.data.frame(data.table::fread(file.path(DATA, "kern_full.csv"), showProgress = FALSE))
}

## Tweet-level human labels: mean rating, majority, and a consensus measure
## (0 = maximal disagreement, 1 = unanimous) over all 15 ratings per tweet.
human_majority <- function(kern = NULL) {
  if (is.null(kern)) kern <- load_kern()
  kern %>%
    group_by(tweet_id) %>%
    summarise(human_OL_mean = mean(offensive_language, na.rm = TRUE),
              human_HS_mean = mean(hate_speech, na.rm = TRUE),
              n_human = dplyr::n(), .groups = "drop") %>%
    mutate(human_OL_maj = as.integer(human_OL_mean >= 0.5),
           human_HS_maj = as.integer(human_HS_mean >= 0.5),
           human_OL_consensus = abs(human_OL_mean - 0.5) * 2,
           human_HS_consensus = abs(human_HS_mean - 0.5) * 2)
}

### ---------------------------------------------------------------- agreement

## Fleiss' kappa for a binary task, from per-item counts of positive ratings.
fleiss_kappa_binary <- function(n_pos, n_raters = 3) {
  n_pos <- as.numeric(n_pos); n <- n_raters
  if (n < 2 || length(n_pos) == 0) return(NA_real_)
  p_bar <- mean(n_pos / n)
  P_i <- (n_pos * (n_pos - 1) + (n - n_pos) * (n - n_pos - 1)) / (n * (n - 1))
  P_e <- p_bar^2 + (1 - p_bar)^2
  if (isTRUE(all.equal(1 - P_e, 0))) return(NA_real_)   # degenerate cell
  (mean(P_i) - P_e) / (1 - P_e)
}

cohens_kappa <- function(a, b) {
  ok <- is.finite(a) & is.finite(b)
  a <- a[ok]; b <- b[ok]
  if (!length(a)) return(NA_real_)
  po <- mean(a == b)
  pe <- mean(a) * mean(b) + (1 - mean(a)) * (1 - mean(b))
  if (isTRUE(all.equal(1 - pe, 0))) return(NA_real_)
  (po - pe) / (1 - pe)
}

## Krippendorff's alpha, nominal binary, items x raters with NA allowed.
## Included so the human numbers can be checked against Kern et al., who
## report alpha rather than kappa.
krippendorff_alpha_binary <- function(m) {
  m <- as.matrix(m)
  Do_num <- 0; Do_den <- 0; vals <- c()
  for (i in seq_len(nrow(m))) {
    r <- m[i, ]; r <- r[is.finite(r)]
    if (length(r) < 2) next
    vals <- c(vals, r)
    Do_num <- Do_num + sum(outer(r, r, "!=")) / (length(r) - 1)
    Do_den <- Do_den + length(r)
  }
  if (Do_den == 0) return(NA_real_)
  Do <- Do_num / Do_den
  n <- length(vals); p1 <- mean(vals)
  De <- 2 * p1 * (1 - p1) * n / (n - 1)
  if (De <= 0) return(NA_real_)
  1 - Do / De
}

## ICC(1) from a one-way random-effects ANOVA on `group`. Unbalanced-safe.
icc_oneway <- function(y, group) {
  ok <- is.finite(y)
  y <- y[ok]; group <- group[ok]
  if (length(y) < 10) return(NA_real_)
  s <- tapply(y, group, sum); cnt <- tapply(y, group, length)
  k <- length(cnt); N <- length(y)
  if (k < 2 || N - k < 1) return(NA_real_)
  means <- s / cnt
  ssb <- sum(cnt * (means - mean(y))^2)
  ssw <- sum(y^2) - sum(s^2 / cnt)
  msb <- ssb / (k - 1); msw <- ssw / (N - k)
  n0 <- (N - sum(cnt^2) / N) / (k - 1)
  va <- (msb - msw) / n0
  if ((va + msw) <= 0) return(NA_real_)
  va / (va + msw)
}

### -------------------------------------------------------------- calibration

## Mean of y within equal-width bins of x. Returns centre, mean, count.
bin_curve <- function(x, y, n_bins = 10, lo = 0, hi = 100) {
  ok <- is.finite(x) & is.finite(y); x <- x[ok]; y <- y[ok]
  edges <- seq(lo, hi, length.out = n_bins + 1)
  out <- lapply(seq_len(n_bins), function(i) {
    sel <- if (i < n_bins) x >= edges[i] & x < edges[i + 1] else x >= edges[i] & x <= edges[i + 1]
    if (!any(sel)) return(NULL)
    data.frame(center = (edges[i] + edges[i + 1]) / 2,
               mean = mean(y[sel]), n = sum(sel))
  })
  do.call(rbind, out)
}

## Expected calibration error; `conf` on the 0-100 scale, `correct` in {0,1}.
ece <- function(conf, correct, n_bins = 10) {
  ok <- is.finite(conf) & is.finite(correct); conf <- conf[ok]; correct <- correct[ok]
  if (!length(conf)) return(NA_real_)
  edges <- seq(0, 100, length.out = n_bins + 1); tot <- 0
  for (i in seq_len(n_bins)) {
    sel <- if (i < n_bins) conf >= edges[i] & conf < edges[i + 1] else conf >= edges[i] & conf <= edges[i + 1]
    if (!any(sel)) next
    tot <- tot + mean(sel) * abs(mean(correct[sel]) - mean(conf[sel]) / 100)
  }
  tot
}

### ------------------------------------------------ instrument variance / deff

## Crossed REML decomposition of run-level prevalence. Each row of `d` is one
## model x condition x run prevalence. Model, condition, and their interaction
## are crossed random effects; the residual is run-to-run variation within a
## model-condition cell. Task-design variance includes the average condition
## effect plus model-specific condition responses. These components are fitted
## jointly, so lower-level variation is not counted again at higher levels.
crossed_prevalence_components <- function(d, n_items = 3000) {
  stopifnot(all(c("model", "condition", "prev") %in% names(d)), nrow(d) > 0)
  d <- d %>%
    mutate(model = factor(model), condition = factor(condition))

  fit <- lme4::lmer(
    prev ~ 1 + (1 | model) + (1 | condition) + (1 | model:condition),
    data = d, REML = TRUE,
    control = lme4::lmerControl(
      optimizer = "bobyqa", optCtrl = list(maxfun = 200000),
      check.conv.singular = "ignore")
  )
  vc <- as.data.frame(lme4::VarCorr(fit))
  variance <- function(group) {
    z <- vc$vcov[vc$grp == group]
    if (length(z) != 1) stop("Expected one variance component for ", group)
    z
  }

  p0 <- mean(d$prev)
  component_var <- c(
    nominal = p0 * (1 - p0) / n_items,
    run = variance("Residual"),
    design = variance("condition") + variance("model:condition"),
    model = variance("model")
  )
  cumulative_var <- cumsum(component_var)

  list(
    fit = fit,
    prevalence = p0,
    component_var = component_var,
    component_sd = sqrt(component_var),
    cumulative_var = cumulative_var,
    cumulative_sd = sqrt(cumulative_var),
    cumulative_deff = cumulative_var / component_var[["nominal"]],
    raw_var = c(
      design = variance("condition"),
      model_design = variance("model:condition"),
      model = variance("model"),
      run = variance("Residual")
    )
  )
}

## Variance inflation relative to the nominal binomial variance at the same n.
## `sds` are additional component standard deviations on the proportion scale.
deff_from_sds <- function(p, n, sds) {
  nominal <- p * (1 - p) / n
  (nominal + sum(sds^2)) / nominal
}

### ------------------------------------------------------------------- cache
###
### The mixed models take minutes on 756,000 rows, so every expensive result is
### memoised to data_work/outputs/cache/. Set PIPELINE_REFRESH=1 in the
### environment to force recomputation (run_all.R does this).
CACHE <- file.path(OUT, "cache")
dir.create(CACHE, showWarnings = FALSE, recursive = TRUE)
REFRESH <- identical(Sys.getenv("PIPELINE_REFRESH"), "1")

cached <- function(name, expr) {
  f <- file.path(CACHE, paste0(name, ".rds"))
  if (!REFRESH && file.exists(f)) {
    cat(sprintf("  [cache hit] %s\n", name))
    return(readRDS(f))
  }
  v <- eval.parent(substitute(expr))
  saveRDS(v, f)
  cat(sprintf("  [computed]  %s\n", name))
  v
}

### ------------------------------------------------------------------ output

## ACL single column is 3.35 in; a figure* spans 6.6 in.
theme_paper <- function(base_size = 6.6) {
  theme_bw(base_size = base_size, base_family = "Helvetica") +
    theme(
      panel.grid.minor = element_blank(),
      panel.grid.major = element_line(linewidth = 0.2, colour = "grey90"),
      panel.border = element_blank(),
      axis.line = element_line(linewidth = 0.3, colour = "black"),
      axis.ticks = element_line(linewidth = 0.3),
      strip.background = element_blank(),
      strip.text = element_text(size = base_size, face = "bold", margin = margin(1, 1, 2, 1)),
      plot.title = element_text(size = base_size, face = "bold", hjust = 0),
      legend.key.size = unit(6, "pt"),
      legend.margin = margin(0, 0, 0, 0),
      legend.box.spacing = unit(2, "pt"),
      legend.background = element_blank(),
      plot.margin = margin(2, 2, 2, 2)
    )
}

## The cairo devices are advertised as available on this machine but fail to
## load (R_X11.so and cairo.so both want libSM / libXrender, which are absent).
## The base pdf() device is therefore the only working vector device. It writes
## Helvetica as one of the 14 standard PDF fonts, so pdflatex embeds nothing and
## the text stays selectable. PNG previews go through quartz.
save_fig <- function(p, name, width, height) {
  pdf_path <- file.path(PLOTS, paste0(name, ".pdf"))
  png_path <- file.path(PLOTS, paste0(name, ".png"))
  grDevices::pdf(pdf_path, width = width, height = height,
                 family = "Helvetica", useDingbats = FALSE)
  print(p); grDevices::dev.off()
  grDevices::png(png_path, width = width, height = height, units = "in",
                 res = 400, type = "quartz")
  print(p); grDevices::dev.off()
  cat(sprintf("  wrote %s (.pdf/.png, %.2f x %.2f in)\n", name, width, height))
  invisible(pdf_path)
}

## Write a bare LaTeX tabular (no float wrapper) that the paper \input's or
## that we paste inline. `lines` is a character vector of tabular body rows.
write_tabular <- function(lines, filename, colspec, header = NULL) {
  body <- c(sprintf("%% auto-generated by %s", basename_or(sys.calls())),
            sprintf("\\begin{tabular}{%s}", colspec),
            "\\toprule",
            if (!is.null(header)) c(header, "\\midrule") else NULL,
            lines,
            "\\bottomrule",
            "\\end{tabular}")
  writeLines(body, file.path(OUT, filename))
  cat(sprintf("  wrote %s\n", filename))
}

basename_or <- function(calls) {
  f <- tryCatch(sys.function(1), error = function(e) NULL)
  s <- tryCatch(basename(sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE)[1])),
                error = function(e) NA)
  if (length(s) && !is.na(s)) s else "R pipeline"
}

## Logger that mirrors to stdout and a per-script log file.
make_logger <- function(logfile) {
  dir.create(file.path(OUT, "r_logs"), showWarnings = FALSE, recursive = TRUE)
  con <- file(file.path(OUT, "r_logs", logfile), open = "wt")
  function(...) {
    s <- paste0(sprintf(...), collapse = "")
    cat(s, "\n", sep = "")
    writeLines(s, con); flush(con)
  }
}

pp <- function(x, digits = 1, signed = TRUE) {
  sprintf(paste0("%", if (signed) "+" else "", ".", digits, "f"), x * 100)
}
