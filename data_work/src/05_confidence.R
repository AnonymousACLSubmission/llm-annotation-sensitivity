################################################################################
### 05_confidence.R  --  does the elicited uncertainty detect any of this?
###
### Six of the twelve eligible conditions ask the model to attach a confidence
### score from 0 to 100 to each label. The question is what that number is a
### claim about, so it is scored against four referents:
###
###   p_run      another run of the same model under the SAME design (reliability)
###   p_design   the same model under a DIFFERENT eligible design    (sensitivity)
###   p_human    a randomly drawn human annotator on the same tweet  (validity)
###   acc_human  the human majority label
###
### A well-behaved score should track all four. It tracks the first, nearly
### tracks the second, and is close to uninformative about the last two.
###
### Outputs
###   plots/confidence_calibration_two_panel.pdf   body figure used in v9
###   plots/confidence_calibration_and_flips.pdf   complete 3-panel figure
###   plots/confidence_calibration_per_model.pdf   appendix figure
###   data_work/outputs/table_confidence_permodel.tex
###
### PARSING NOTE. The confidence scores must be aligned to the label token they
### accompany, not to a fixed position. Our task_structure B lists hate speech
### FIRST, so positional assignment silently swaps both of its scores. Aligning
### on the token (OL/NO for offensive language, HS/NH for hate speech) is
### order-agnostic and correct in every condition.
################################################################################

BASEDIR <- "/Users/sseckman/Downloads/Annotation_Sensitivity_in_Large_Language_Models__ACL_ (1)"
source(file.path(BASEDIR, "data_work", "src", "00_utils.R"))
suppressPackageStartupMessages({
  library(purrr); library(patchwork); library(pROC); library(data.table)
})

say <- make_logger("05_confidence.log")

OUTCOMES     <- c("OL", "HS")
OUTCOME_NAME <- c(OL = "Offensive language", HS = "Hate speech")
OL_TOKENS    <- c("OL", "NO")
HS_TOKENS    <- c("HS", "NH")
TARGETS      <- c(p_run = "another run, same design",
                  p_design = "same model, different design",
                  p_human = "a random human annotator",
                  acc_human = "the human majority label")

### ------------------------------------------------------------------- parsing

## The label and score columns hold Python list literals, e.g. "['HS', 'OL']"
## and "['0', '100']". Strip the brackets and quotes and split on commas.
split_literal <- function(x) {
  x <- gsub("^\\s*\\[|\\]\\s*$", "", x)
  strsplit(x, "\\s*,\\s*")
}

parse_scores <- function(label_str, score_str) {
  toks <- lapply(split_literal(label_str),
                 function(v) toupper(trimws(gsub("['\"]", "", v))))
  nums <- lapply(split_literal(score_str),
                 function(v) suppressWarnings(as.numeric(gsub("['\"]", "", v))))
  n <- length(toks)
  c_ol <- rep(NA_real_, n); c_hs <- rep(NA_real_, n)
  for (i in seq_len(n)) {
    t <- toks[[i]]; v <- nums[[i]]
    if (length(t) != length(v)) next
    keep <- is.finite(v) & v >= 0 & v <= 100
    if (!any(keep)) next
    t <- t[keep]; v <- v[keep]
    if (any(t %in% OL_TOKENS)) c_ol[i] <- v[which(t %in% OL_TOKENS)[1]]
    if (any(t %in% HS_TOKENS)) c_hs[i] <- v[which(t %in% HS_TOKENS)[1]]
  }
  list(OL = c_ol, HS = c_hs)
}

### --------------------------------------------------------------------- data

say("Loading df_agg ...")
agg <- load_agg() %>%
  select(tweet_id, condition, model, task_structure, batched, confidence,
         modal_OL, modal_HS)

say("Loading df_long, confidence conditions only ...")
long <- load_long(cols = c("tweet_id", "condition", "model", "task_structure",
                           "batched", "confidence", "responder", "label",
                           "score", "OL", "HS")) %>%
  filter(confidence == 1)
say("  confidence rows: %s", format(nrow(long), big.mark = ","))

p <- parse_scores(long$label, long$score)
long$conf_OL <- p$OL
long$conf_HS <- p$HS

say("\nParse check (OL score first in A, HS score first in B):")
for (ts in c("A", "B", "C.OL", "C.HS")) {
  r <- long[long$task_structure == ts, ][1, ]
  say("  %-5s label=%-16s score=%-16s -> conf_OL=%s conf_HS=%s", ts, r$label,
      r$score, r$conf_OL, r$conf_HS)
}
for (oc in OUTCOMES) {
  e <- eligible(long, oc)
  say("  valid conf_%s: %s / %s eligible rows", oc,
      format(sum(!is.na(e[[paste0("conf_", oc)]])), big.mark = ","),
      format(nrow(e), big.mark = ","))
}
say("  total scored labels: %s",
    format(sum(!is.na(eligible(long, "OL")$conf_OL)) +
           sum(!is.na(eligible(long, "HS")$conf_HS)), big.mark = ","))

hum <- human_majority()

### ------------------------------------- item-level instability across designs

say("\n%s\nITEM-LEVEL INSTABILITY ACROSS THE TWELVE DESIGNS\n%s",
    strrep("=", 78), strrep("=", 78))

inst <- lapply(OUTCOMES, function(oc) {
  e <- eligible(agg, oc)
  g <- e %>% group_by(model, tweet_id) %>%
    summarise(n_pos = sum(.data[[paste0("modal_", oc)]], na.rm = TRUE),
              n_designs = dplyr::n(), .groups = "drop") %>%
    ## Dissent is the size of the minority position, so it is symmetric in the
    ## label and maximal at an even split.
    mutate(n_dissent = pmin(n_pos, n_designs - n_pos),
           instability = n_dissent / n_designs,
           flips = as.integer(n_dissent > 0))
  say("\n%s: designs per item = %s", oc,
      paste(sort(unique(g$n_designs)), collapse = ", "))
  say("  share of (model, tweet) pairs that flip at least once: %.1f%%",
      mean(g$flips) * 100)
  say("  mean instability index (0 = unanimous, .5 = maximal): %.3f",
      mean(g$instability))
  say("  by model:")
  for (m in MODEL_ORDER) {
    s <- g[g$model == m, ]
    say("    %-20s flip %5.1f%%   instability %.3f", m, mean(s$flips) * 100,
        mean(s$instability))
  }
  g
})
names(inst) <- OUTCOMES

## The same statistic without the eligibility filter, i.e. the bug this pipeline
## fixes, reported so the correction is auditable.
say("\nWithout the eligibility filter the flip rates would be:")
for (oc in OUTCOMES) {
  g <- agg %>% group_by(model, tweet_id) %>%
    summarise(k = dplyr::n_distinct(.data[[paste0("modal_", oc)]]), .groups = "drop")
  say("  %s: %.1f%%", oc, mean(g$k > 1) * 100)
}

### --------------------------------------- confidence as a prediction of agreement

say("\n%s\nCONFIDENCE AS A PREDICTION OF AGREEMENT\n%s", strrep("=", 78),
    strrep("=", 78))

build <- function(oc) {
  e <- eligible(long, oc)
  e <- e[!is.na(e[[paste0("conf_", oc)]]) & !is.na(e[[oc]]), ]
  lab <- as.numeric(e[[oc]])

  ## p_run: leave-one-out over the runs of the same model x condition x tweet.
  d <- as.data.table(e)
  d[, `:=`(.csum = sum(get(oc), na.rm = TRUE), .cn = .N),
    by = .(model, condition, tweet_id)]
  other_pos <- d$.csum - lab
  other_n   <- d$.cn - 1
  p_run <- ifelse(lab == 1, other_pos, other_n - other_pos) / other_n

  ## p_design: leave-one-out over the twelve designs' modal labels.
  e <- e %>%
    left_join(inst[[oc]][, c("model", "tweet_id", "n_pos", "n_designs",
                             "instability", "flips")],
              by = c("model", "tweet_id")) %>%
    left_join(agg %>% select(model, condition, tweet_id,
                             own_modal = all_of(paste0("modal_", oc))),
              by = c("model", "condition", "tweet_id"))
  opos <- e$n_pos - e$own_modal
  on_  <- e$n_designs - 1
  p_design <- ifelse(lab == 1, opos, on_ - opos) / on_

  ## p_human: the share of human annotators who would give this same label.
  e <- e %>% left_join(hum %>% select(tweet_id,
                                      hmean = all_of(paste0("human_", oc, "_mean")),
                                      hmaj  = all_of(paste0("human_", oc, "_maj")),
                                      hcons = all_of(paste0("human_", oc, "_consensus"))),
                       by = "tweet_id")

  tibble::tibble(model = e$model, condition = e$condition,
                 task_structure = e$task_structure, batched = e$batched,
                 tweet_id = e$tweet_id, outcome = oc,
                 conf = e[[paste0("conf_", oc)]],
                 p_run = p_run, p_design = p_design,
                 p_human = ifelse(lab == 1, e$hmean, 1 - e$hmean),
                 acc_human = as.numeric(lab == e$hmaj),
                 instability = e$instability, flips = e$flips,
                 consensus = e$hcons)
}

D <- map_dfr(OUTCOMES, build)
say("\nAnalysis rows: %s", format(nrow(D), big.mark = ","))

for (oc in OUTCOMES) {
  d <- D[D$outcome == oc, ]
  say("\n--- %s (pooled over 7 models, %s labels) ---", oc,
      format(nrow(d), big.mark = ","))
  say("  mean stated confidence: %.1f", mean(d$conf, na.rm = TRUE))
  say("  %-34s%8s%10s%8s", "target", "mean", "gap (pp)", "ECE")
  for (k in names(TARGETS)) {
    m <- mean(d[[k]], na.rm = TRUE)
    say("  %-34s%7.1f%%%10.1f%8.3f", TARGETS[[k]], m * 100,
        mean(d$conf, na.rm = TRUE) - m * 100, ece(d$conf, d[[k]]))
  }
}

### ------------------------------------------- does low confidence mark flips?

say("\n%s\nDOES LOW CONFIDENCE MARK THE ITEMS THAT FLIP?\n%s", strrep("=", 78),
    strrep("=", 78))

item <- D %>% group_by(model, outcome, tweet_id) %>%
  summarise(conf = mean(conf, na.rm = TRUE), instability = first(instability),
            flips = first(flips), consensus = first(consensus), .groups = "drop")

auc_flip <- function(d) {
  if (dplyr::n_distinct(d$flips) < 2) return(NA_real_)
  as.numeric(pROC::auc(pROC::roc(d$flips, -d$conf, quiet = TRUE,
                                 direction = "<", levels = c(0, 1))))
}

for (oc in OUTCOMES) {
  d <- item[item$outcome == oc & is.finite(item$conf) & !is.na(item$flips), ]
  rho <- suppressWarnings(cor(d$conf, d$instability, method = "spearman"))
  say("\n%s: item-level AUC(flip | low conf) = %.3f   Spearman(conf, instability) = %+.3f   n = %s",
      oc, auc_flip(d), rho, format(nrow(d), big.mark = ","))
  hi <- d[d$conf >= 95, ]
  say("   items given >= 95 confidence: %.1f%% of items, and %.1f%% of them still flip",
      nrow(hi) / nrow(d) * 100, mean(hi$flips) * 100)
  bc <- bin_curve(d$conf, d$flips, n_bins = 10)
  say("   flip rate by confidence bin: %s",
      paste(sprintf("%d:%.0f%%", round(bc$center), bc$mean * 100), collapse = "  "))
  ## Confident items are numerous, so a low per-item flip rate can still account
  ## for most of the flips.
  fl <- d[d$flips == 1, ]
  for (thr in c(90, 95))
    say("   of all items that flip, %.1f%% were given mean confidence >= %d",
        mean(fl$conf >= thr) * 100, thr)
  say("   mean confidence: stable items %.1f vs flipping items %.1f",
      mean(d$conf[d$flips == 0]), mean(fl$conf))
}

### ------------------------------------ the uncertainty is itself design-dependent

say("\n%s\nTHE UNCERTAINTY ESTIMATE IS ITSELF DESIGN-DEPENDENT\n%s",
    strrep("=", 78), strrep("=", 78))
for (oc in OUTCOMES) {
  d <- D[D$outcome == oc, ]
  t <- d %>% group_by(condition) %>%
    summarise(conf = mean(conf, na.rm = TRUE), .groups = "drop") %>% arrange(conf)
  say("\n%s mean confidence by condition:", oc)
  for (i in seq_len(nrow(t))) say("   %-20s %6.2f", t$condition[i], t$conf[i])
  say("   range across designs: %.2f points", max(t$conf) - min(t$conf))
  say("   per-model range across designs:")
  for (m in MODEL_ORDER) {
    tm <- d %>% filter(model == m) %>% group_by(condition) %>%
      summarise(c = mean(conf, na.rm = TRUE), .groups = "drop")
    say("     %-20s %6.2f - %6.2f  (range %5.2f)", m, min(tm$c), max(tm$c),
        max(tm$c) - min(tm$c))
  }
  b0 <- mean(d$conf[d$batched == 0], na.rm = TRUE)
  b1 <- mean(d$conf[d$batched == 1], na.rm = TRUE)
  say("   batched 0: %.2f   1: %.2f   diff %+.2f", b0, b1, b1 - b0)
  for (ts in sort(unique(d$task_structure)))
    say("     structure %-6s %.2f", ts,
        mean(d$conf[d$task_structure == ts], na.rm = TRUE))
  say("     per-model batch effect (batch minus individual):")
  for (m in MODEL_ORDER) {
    dm <- d[d$model == m, ]
    say("       %-20s %+6.2f", m,
        mean(dm$conf[dm$batched == 1], na.rm = TRUE) -
        mean(dm$conf[dm$batched == 0], na.rm = TRUE))
  }
}

### ------------------------------------------------ robustness: human consensus

say("\n%s\nROBUSTNESS: ITEMS WITH HIGH HUMAN CONSENSUS ONLY (>= 0.6)\n%s",
    strrep("=", 78), strrep("=", 78))
for (oc in OUTCOMES) {
  d <- D[D$outcome == oc & D$consensus >= 0.6, ]
  say("\n%s: %s labels on high-consensus items", oc,
      format(nrow(d), big.mark = ","))
  for (k in names(TARGETS))
    say("   %-34s%7.1f%%   ECE %.3f", TARGETS[[k]],
        mean(d[[k]], na.rm = TRUE) * 100, ece(d$conf, d[[k]]))
}

### ------------------------------------------------------------ per-model table

say("\n%s\nPER-MODEL CALIBRATION (tab:confidence_permodel)\n%s", strrep("=", 78),
    strrep("=", 78))

T <- map_dfr(OUTCOMES, function(oc) map_dfr(MODEL_ORDER, function(m) {
  d <- D[D$outcome == oc & D$model == m, ]
  i <- item[item$outcome == oc & item$model == m &
            is.finite(item$conf) & !is.na(item$flips), ]
  tibble::tibble(outcome = oc, model = m,
                 ece_run = ece(d$conf, d$p_run),
                 ece_design = ece(d$conf, d$p_design),
                 ece_human = ece(d$conf, d$p_human),
                 conf = mean(d$conf, na.rm = TRUE),
                 flip = mean(i$flips),
                 auc = auc_flip(i),
                 rho = suppressWarnings(cor(i$conf, i$instability,
                                            method = "spearman")))
}))
readr::write_csv(T, file.path(OUT, "confidence_per_model.csv"))
for (i in seq_len(nrow(T)))
  say("%-3s %-20s ECE run %.3f design %.3f human %.3f | conf %.1f | flip %.1f%% | AUC %.3f",
      T$outcome[i], T$model[i], T$ece_run[i], T$ece_design[i], T$ece_human[i],
      T$conf[i], T$flip[i] * 100, T$auc[i])

lines <- unlist(lapply(OUTCOMES, function(oc) c(
  sprintf("\\multicolumn{7}{l}{\\emph{%s}} \\\\", OUTCOME_NAME[[oc]]),
  vapply(MODEL_ORDER, function(m) {
    r <- T[T$outcome == oc & T$model == m, ]
    sprintf("\\quad %s & %.3f & %.3f & %.3f & %.1f & %.1f\\%% & %.3f \\\\",
            MODEL_LABELS[[m]], r$ece_run, r$ece_design, r$ece_human, r$conf,
            r$flip * 100, r$auc)
  }, character(1)))))

write_tabular(lines, "table_confidence_permodel.tex", "lrrrrrr", header = c(
  "& \\multicolumn{3}{c}{ECE against agreement with\\ldots} & & & \\\\",
  "\\cmidrule(lr){2-4}",
  paste("Model & another run & another design & a human & Mean conf. &",
        "Flip rate & AUC \\\\")))

### ================================================================== figures

REF     <- c("p_run", "p_design", "p_human")
REF_LAB <- c(p_run = "another run, same design",
             p_design = "same model, other design",
             p_human = "a random human")
REF_LTY <- c(p_run = "solid", p_design = "dashed", p_human = "dotted")
REF_SHP <- c(p_run = 16, p_design = 15, p_human = 17)
OC_COL  <- c(OL = OL_COLOR, HS = HS_COLOR)

## Calibration curves: mean agreement within ten equal-width bins of stated
## confidence, one line per referent and outcome.
curves <- function(dat, n_bins = 10) {
  map_dfr(OUTCOMES, function(oc) map_dfr(REF, function(k) {
    d <- dat[dat$outcome == oc, ]
    bc <- bin_curve(d$conf, d[[k]], n_bins = n_bins)
    if (is.null(bc)) return(NULL)
    tibble::tibble(outcome = oc, ref = k, x = bc$center / 100, y = bc$mean)
  })) %>%
    mutate(ref = factor(ref, levels = REF),
           outcome = factor(outcome, levels = OUTCOMES))
}

cal <- curves(D)

panel_a <- ggplot(cal, aes(x, y, colour = outcome, linetype = ref, shape = ref)) +
  geom_abline(slope = 1, intercept = 0, colour = "grey40", linewidth = 0.25,
              linetype = "dotted") +
  annotate("text", x = 0.365, y = 0.315, label = "stated = actual", angle = 42,
           size = 1.85, colour = "grey40", hjust = 0, vjust = 0) +
  geom_line(linewidth = 0.34) + geom_point(size = 0.75) +
  scale_colour_manual(values = OC_COL, labels = OUTCOME_NAME, name = NULL) +
  scale_linetype_manual(values = REF_LTY, labels = REF_LAB, name = NULL) +
  scale_shape_manual(values = REF_SHP, labels = REF_LAB, name = NULL) +
  guides(linetype = guide_legend(order = 1), shape = guide_legend(order = 1),
         colour = guide_legend(order = 2)) +
  coord_cartesian(xlim = c(0.28, 1.02), ylim = c(0.28, 1.02)) +
  labs(title = "(a) Calibrated to itself, not to humans",
       x = "Stated confidence", y = "P(another labeler agrees)") +
  theme_paper(6.6) +
  theme(legend.position = "inside", legend.position.inside = c(0.02, 0.99),
        legend.justification = c(0, 1), legend.spacing.y = unit(0, "pt"),
        legend.text = element_text(size = 4.6), legend.key.width = unit(13, "pt"),
        legend.key.height = unit(5.5, "pt"))

## Panel b: flip rate by binned item confidence, with the overall rate as a
## reference line.
flipcurve <- map_dfr(OUTCOMES, function(oc) {
  d <- item[item$outcome == oc & is.finite(item$conf) & !is.na(item$flips), ]
  bc <- bin_curve(d$conf, d$flips, n_bins = 10)
  tibble::tibble(outcome = oc, x = bc$center / 100, y = bc$mean,
                 base = mean(d$flips))
})
flipcurve <- flipcurve %>% mutate(outcome = factor(outcome, levels = OUTCOMES))
base_lines <- flipcurve %>% group_by(outcome) %>%
  summarise(base = first(base), .groups = "drop")

panel_b <- ggplot(flipcurve, aes(x, y, colour = outcome)) +
  geom_hline(data = base_lines, aes(yintercept = base, colour = outcome),
             linewidth = 0.22, linetype = "longdash", alpha = 0.85) +
  geom_text(data = base_lines, aes(x = 0.30, y = base + 0.03,
                                   label = sprintf("%s overall %.0f%%", outcome,
                                                   base * 100)),
            size = 1.85, hjust = 0, show.legend = FALSE) +
  geom_line(linewidth = 0.34) + geom_point(size = 0.75) +
  scale_colour_manual(values = OC_COL, labels = OUTCOME_NAME, name = NULL) +
  coord_cartesian(xlim = c(0.28, 1.02), ylim = c(0, 1.04)) +
  labs(title = "(b) Low confidence flags flip-prone items",
       x = "Mean stated confidence for the item",
       y = "Share of items whose label\nflips across the 12 designs") +
  theme_paper(6.6) +
  theme(legend.position = "inside", legend.position.inside = c(0.02, 0.02),
        legend.justification = c(0, 0),
        legend.text = element_text(size = 5.0))

## Panel c: mean confidence under individual and batched presentation, one line
## per model, with the pooled shift drawn behind as a thick arrow.
slopes <- map_dfr(OUTCOMES, function(oc) map_dfr(MODEL_ORDER, function(m) {
  d <- D[D$outcome == oc & D$model == m, ]
  tibble::tibble(outcome = oc, model = m,
                 y0 = mean(d$conf[d$batched == 0], na.rm = TRUE),
                 y1 = mean(d$conf[d$batched == 1], na.rm = TRUE))
}))
pooled <- map_dfr(OUTCOMES, function(oc) {
  d <- D[D$outcome == oc, ]
  tibble::tibble(outcome = oc,
                 y0 = mean(d$conf[d$batched == 0], na.rm = TRUE),
                 y1 = mean(d$conf[d$batched == 1], na.rm = TRUE))
})
XS <- c(OL = 1, HS = 2)
slopes <- slopes %>% mutate(x0 = XS[outcome] - 0.19, x1 = XS[outcome] + 0.19,
                            model = factor(model, levels = MODEL_ORDER))
pooled <- pooled %>% mutate(x0 = XS[outcome] - 0.19, x1 = XS[outcome] + 0.19)

panel_c <- ggplot() +
  geom_segment(data = pooled, aes(x = x0, xend = x1, y = y0, yend = y1,
                                  colour = outcome),
               linewidth = 1.1, alpha = 0.3, show.legend = FALSE) +
  geom_segment(data = slopes, aes(x = x0, xend = x1, y = y0, yend = y1,
                                  colour = outcome),
               linewidth = 0.28, alpha = 0.8, show.legend = FALSE) +
  geom_point(data = slopes, aes(x = x0, y = y0, colour = outcome, shape = model,
                               fill = outcome), size = 0.9) +
  geom_point(data = slopes, aes(x = x1, y = y1, colour = outcome, shape = model),
             size = 0.9, fill = "white") +
  scale_colour_manual(values = OC_COL, guide = "none") +
  scale_fill_manual(values = OC_COL, guide = "none") +
  scale_shape_manual(values = MODEL_SHAPE[MODEL_ORDER], guide = "none") +
  scale_x_continuous(breaks = c(0.81, 1.19, 1.81, 2.19),
                     labels = rep(c("one\nat a time", "in a\nbatch"), 2),
                     sec.axis = dup_axis(breaks = c(1, 2),
                                         labels = OUTCOME_NAME[c("OL", "HS")],
                                         name = NULL),
                     limits = c(0.55, 2.45)) +
  coord_cartesian(ylim = c(48, 100)) +
  labs(title = "(c) Confidence moves with the design", x = NULL,
       y = "Mean stated confidence") +
  theme_paper(6.6) +
  theme(panel.grid.major.x = element_blank(),
        axis.text.x.bottom = element_text(size = 4.4),
        axis.text.x.top = element_text(size = 5.6, face = "bold",
                                       colour = c(OL_COLOR, HS_COLOR)),
        axis.ticks.x.top = element_blank(),
        legend.position = "none")

save_fig(panel_a | panel_b | panel_c, "confidence_calibration_and_flips",
         width = 6.6, height = 2.35)
## The current manuscript uses calibration and design-dependent confidence only.
## Keep the three-panel version above as a complete analysis artifact and write
## this exact two-panel asset for the body figure.
save_fig(panel_a | panel_c, "confidence_calibration_two_panel",
         width = 4.6, height = 2.35)

### ----------------------------------------------------------- appendix figure

per_model_cal <- map_dfr(MODEL_ORDER, function(m) {
  d <- D[D$model == m, ]
  curves(d, n_bins = 8) %>% mutate(model = m)
}) %>% mutate(model = factor(model, levels = MODEL_ORDER,
                             labels = MODEL_LABELS[MODEL_ORDER]))

p_app <- ggplot(per_model_cal, aes(x, y, colour = outcome, linetype = ref,
                                   shape = ref)) +
  geom_abline(slope = 1, intercept = 0, colour = "grey45", linewidth = 0.22,
              linetype = "dotted") +
  geom_line(linewidth = 0.3) + geom_point(size = 0.6) +
  facet_wrap(~ model, ncol = 4) +
  scale_colour_manual(values = OC_COL, labels = OUTCOME_NAME, name = NULL) +
  scale_linetype_manual(values = REF_LTY, labels = REF_LAB, name = NULL) +
  scale_shape_manual(values = REF_SHP, labels = REF_LAB, name = NULL) +
  guides(linetype = guide_legend(order = 1), shape = guide_legend(order = 1),
         colour = guide_legend(order = 2)) +
  coord_cartesian(xlim = c(0.28, 1.02), ylim = c(0.28, 1.02)) +
  labs(x = "Stated confidence", y = "P(another labeler agrees)") +
  theme_paper(6.0) +
  ## The legend lives in the empty eighth facet slot. Anchoring its right edge
  ## to the right edge of the plot is what keeps the labels from being clipped.
  theme(legend.position = "inside", legend.position.inside = c(1, 0.22),
        legend.justification = c(1, 0.5),
        legend.text = element_text(size = 4.2),
        legend.key.width = unit(12, "pt"),
        legend.key.height = unit(5, "pt"),
        legend.spacing.y = unit(0, "pt"),
        strip.text = element_text(size = 5.4, face = "plain"),
        axis.text = element_text(size = 4.4),
        panel.spacing = unit(2.5, "pt"))

save_fig(p_app, "confidence_calibration_per_model", width = 3.35, height = 2.1)

say("\nDone.")
