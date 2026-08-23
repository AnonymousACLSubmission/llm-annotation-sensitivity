################################################################################
### 06_prevalence.R  --  descriptive prevalence figures
###
### Two appendix figures, both built from run-level prevalences (one value per
### model x condition x run):
###
###   prevalence_heatmap_all_models_vs_model_mean_stacked
###     Each cell is a condition's prevalence minus that model's own mean over
###     its twelve eligible conditions. Removing the model mean is what makes the
###     design effect visible: a column that is uniformly warm or cool is a model
###     shift, a row that is uniformly warm or cool is a design shift.
###
###   prevalence_stability_runs
###     The three runs of every cell, drawn as three points with the cell mean as
###     a bar. Runs of one cell are nearly indistinguishable; cells of one model
###     are far apart. This is the reliability/sensitivity dissociation in raw
###     form, before any model is fitted.
################################################################################

BASEDIR <- "/Users/sseckman/Downloads/Annotation_Sensitivity_in_Large_Language_Models__ACL_ (1)"
source(file.path(BASEDIR, "data_work", "src", "00_utils.R"))
suppressPackageStartupMessages({ library(purrr); library(patchwork) })

say <- make_logger("06_prevalence.log")

OUTCOMES     <- c("OL", "HS")
OUTCOME_NAME <- c(OL = "Offensive language", HS = "Hate speech")
VARIANTS     <- c("", "_conf", "_batch", "_batch_conf")
VAR_LABEL    <- c("Base", "+ Confidence", "Batch", "Batch + Confidence")
STRUCTS      <- list(OL = c("A", "B", "C.OL"), HS = c("A", "B", "C.HS"))
STRUCT_LAB   <- c("Joint (OL first)", "Joint (HS first)", "Separate")

## Position 1..12 inside a panel, shared by both outcomes: the twelve cells are
## the three task structures crossed with the four prompt variants.
slot_table <- function(oc) {
  data.frame(condition = as.vector(outer(VARIANTS, STRUCTS[[oc]],
                                         function(v, s) paste0(s, v))),
             slot = seq_len(12),
             struct = rep(STRUCT_LAB, each = 4),
             variant = rep(VAR_LABEL, 3),
             stringsAsFactors = FALSE)
}

### ------------------------------------------------------- run-level prevalence

say("Loading run-level labels ...")
long <- load_long(cols = c("tweet_id", "condition", "model", "task_structure",
                           "responder", "OL", "HS"))
say("  rows: %s", format(nrow(long), big.mark = ","))

run_prev <- map_dfr(OUTCOMES, function(oc) {
  eligible(long, oc) %>%
    group_by(model, condition, responder) %>%
    ## Exactly one of the 756,000 eligible responses failed to parse (a
    ## Llama 3.1 8B call in the joint OL-first condition), which is why the
    ## paper counts 755,999 labels. na.rm drops it.
    summarise(prev = mean(.data[[oc]], na.rm = TRUE),
              n = sum(!is.na(.data[[oc]])), .groups = "drop") %>%
    mutate(outcome = oc) %>%
    left_join(slot_table(oc), by = "condition")
})
stopifnot(all(!is.na(run_prev$slot)), all(run_prev$n >= 2999), all(run_prev$n <= 3000))
say("  labels used: %s (of %s requested)", format(sum(run_prev$n), big.mark = ","),
    format(nrow(run_prev) * 3000, big.mark = ","))
say("  run-level cells: %d (%d per outcome)", nrow(run_prev), nrow(run_prev) / 2)
write.csv(run_prev, file.path(OUT, "run_level_prevalence.csv"), row.names = FALSE)

cell_prev <- run_prev %>%
  group_by(outcome, model, condition, slot, struct, variant) %>%
  summarise(sd_run = sd(prev), prev = mean(prev), .groups = "drop")

### --------------------------------------------------- figure: deviation heatmap

dev_tab <- cell_prev %>%
  group_by(outcome, model) %>%
  mutate(model_mean = mean(prev), dev = (prev - model_mean) * 100) %>%
  ungroup() %>%
  mutate(model = factor(model, levels = MODEL_ORDER,
                        labels = MODEL_LABELS[MODEL_ORDER]),
         row = factor(slot, levels = 12:1),
         lab = sprintf("%+d", round(dev)))

lim <- max(abs(dev_tab$dev))
say("\nLargest deviation from a model's own mean: %.1f pp", lim)
for (oc in OUTCOMES) {
  d <- dev_tab[dev_tab$outcome == oc, ]
  say("  %s: SD of the twelve deviations, by model:", oc)
  for (m in levels(d$model)) {
    say("    %-20s %.2f pp", m, sd(d$dev[d$model == m]))
  }
}

row_labels <- slot_table("OL")[order(slot_table("OL")$slot), ]
row_labels$lab <- paste0(row_labels$struct, "\n", row_labels$variant)

heat_panel <- function(oc, with_x) {
  d <- dev_tab[dev_tab$outcome == oc, ]
  g <- ggplot(d, aes(x = model, y = row, fill = dev)) +
    geom_tile(colour = "white", linewidth = 0.4) +
    geom_text(aes(label = lab, colour = abs(dev) > 0.72 * lim), size = 1.5,
              show.legend = FALSE) +
    scale_colour_manual(values = c(`TRUE` = "white", `FALSE` = "grey15"),
                        guide = "none") +
    scale_fill_distiller(palette = "RdBu", direction = -1,
                         limits = c(-lim, lim),
                         name = "Deviation from model's own mean (pp)") +
    scale_y_discrete(labels = setNames(row_labels$lab, row_labels$slot),
                     expand = expansion(0, 0)) +
    scale_x_discrete(expand = expansion(0, 0), position = "bottom") +
    labs(title = OUTCOME_NAME[[oc]]) +
    theme_paper(6) +
    theme(panel.grid = element_blank(), axis.line = element_blank(),
          axis.ticks = element_blank(), axis.title = element_blank(),
          axis.text.y = element_text(size = 3.9, lineheight = 0.9, hjust = 1),
          axis.text.x = if (with_x)
            element_text(size = 4.4, angle = 45, hjust = 1) else element_blank(),
          plot.title = element_text(size = 6, face = "bold", hjust = 0.5),
          legend.position = "bottom", legend.direction = "horizontal",
          legend.title = element_text(size = 5.2),
          legend.text = element_text(size = 4.4),
          legend.key.height = unit(4, "pt"), legend.key.width = unit(26, "pt"),
          plot.margin = margin(2, 2, 2, 2)) +
    guides(fill = guide_colourbar(title.position = "top", title.hjust = 0.5,
                                 ticks.colour = "white"))
  g
}

p_heat <- (heat_panel("OL", TRUE) / heat_panel("HS", TRUE)) +
  plot_layout(guides = "collect") & theme(legend.position = "bottom")
save_fig(p_heat, "prevalence_heatmap_all_models_vs_model_mean_stacked",
         width = 3.35, height = 5.2)

### ------------------------------------------------- figure: run-level stability

## The right-hand facet strips run vertically, so a strip label can only be as
## long as a panel is tall. Seven panels in 3.5 in leaves room for roughly
## thirteen characters, so the two longest names wrap onto two lines rather than
## being shrunk further.
STAB_LABELS <- c("GPT-4o-mini" = "GPT-4o-mini", "GPT-5.4" = "GPT-5.4",
                 "Mistral-Large-3" = "Mistral\nLarge 3",
                 "Mistral-Medium-3.5" = "Mistral\nMedium 3.5",
                 "Llama-3.1-8B" = "Llama 3.1\n8B",
                 "Llama-3.1-70B" = "Llama 3.1\n70B",
                 "Llama-4" = "Llama 4")

stab <- run_prev %>%
  mutate(family = MODEL_FAM[model],
         model = factor(model, levels = MODEL_ORDER,
                        labels = STAB_LABELS[MODEL_ORDER]),
         prev = prev * 100)

## Offset the three runs by a fixed amount rather than jittering at random, so
## that the figure is byte-identical on every run of the script.
stab <- stab %>% group_by(outcome, model, slot) %>%
  mutate(x = slot + c(-0.24, 0, 0.24)[rank(responder, ties.method = "first")]) %>%
  ungroup()
means <- stab %>% group_by(outcome, model, slot) %>%
  summarise(prev = mean(prev), .groups = "drop")

X_LAB <- rep(c("base", "conf", "batch", "b+c"), 3)

stab_col <- function(oc, ylab) {
  d  <- stab[stab$outcome == oc, ]
  mn <- means[means$outcome == oc, ]
  ggplot(d, aes(x = x, y = prev, colour = family)) +
    geom_vline(xintercept = c(4.5, 8.5), colour = "grey88", linewidth = 0.2) +
    geom_point(size = 0.3, alpha = 0.9) +
    geom_segment(data = mn, aes(x = slot - 0.36, xend = slot + 0.36,
                                y = prev, yend = prev),
                 colour = "black", linewidth = 0.25, alpha = 0.8,
                 inherit.aes = FALSE) +
    facet_wrap(~ model, ncol = 1, strip.position = "right") +
    scale_colour_manual(values = FAM_COLORS, guide = "none") +
    scale_x_continuous(breaks = 1:12, labels = X_LAB,
                       sec.axis = dup_axis(breaks = c(2.5, 6.5, 10.5),
                                           labels = c("Joint, OL first",
                                                      "Joint, HS first",
                                                      "Separate"), name = NULL),
                       limits = c(0.4, 12.6), expand = expansion(0, 0)) +
    labs(title = OUTCOME_NAME[[oc]], x = NULL,
         y = if (ylab) "Estimated prevalence (%)" else NULL) +
    theme_paper(6) +
    theme(panel.grid.major.x = element_blank(),
          plot.title = element_text(size = 5.8, face = "bold", hjust = 0.5),
          axis.text.x.bottom = element_text(size = 3.5, angle = 90, hjust = 1,
                                            vjust = 0.5),
          axis.text.x.top = element_text(size = 4.2),
          axis.ticks.x.top = element_blank(),
          axis.text.y = element_text(size = 4.0),
          axis.title.y = element_text(size = 5.4),
          strip.text.y.right = element_text(size = 3.6, angle = -90, lineheight = 0.9,
                                            margin = margin(1, 1.5, 1, 1.5)),
          panel.spacing.y = unit(1.6, "pt"),
          plot.margin = margin(2, 2, 2, 2))
}

p_stab <- stab_col("OL", TRUE) | stab_col("HS", FALSE)
save_fig(p_stab, "prevalence_stability_runs", width = 3.35, height = 3.5)

### ------------------------------------------- within versus across, in numbers

say("\n%s\nWITHIN-CELL SD (reliability) VERSUS ACROSS-CELL SD (sensitivity)\n%s",
    strrep("=", 78), strrep("=", 78))
say("%-20s %-24s %-24s", "model", "OL within / across", "HS within / across")
for (m in MODEL_ORDER) {
  parts <- vapply(OUTCOMES, function(oc) {
    r <- run_prev[run_prev$model == m & run_prev$outcome == oc, ]
    w <- r %>% group_by(condition) %>% summarise(s = sd(prev), .groups = "drop")
    a <- r %>% group_by(condition) %>% summarise(p = mean(prev), .groups = "drop")
    sprintf("%.2f / %.2f pp  (%.0fx)", mean(w$s) * 100, sd(a$p) * 100,
            sd(a$p) / mean(w$s))
  }, character(1))
  say("%-20s %-24s %-24s", m, parts[["OL"]], parts[["HS"]])
}

say("\nPrevalence by model, pooled over the twelve eligible conditions:")
for (oc in OUTCOMES) {
  mm <- cell_prev %>% filter(outcome == oc) %>% group_by(model) %>%
    summarise(p = mean(prev), .groups = "drop")
  say("  %s: %s", oc, paste(sprintf("%s %.1f%%", mm$model, mm$p * 100),
                            collapse = "  "))
  say("     range across models %.1f - %.1f%%", min(mm$p) * 100, max(mm$p) * 100)
  cellr <- cell_prev %>% filter(outcome == oc)
  say("     range across all %d model x condition cells %.1f - %.1f%%",
      nrow(cellr), min(cellr$prev) * 100, max(cellr$prev) * 100)
}
say("\nDone.")
