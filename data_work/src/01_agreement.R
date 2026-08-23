################################################################################
### 01_agreement.R  --  the agreement ladder
###
### Three rungs, all on the same 3,000 items:
###   within a design   Fleiss' kappa over the three runs of one model x condition
###   across designs    Cohen's kappa between the modal labels of two conditions
###                     of the same model
###   against humans    Cohen's kappa between a condition's modal label and the
###                     majority of the 15 human ratings in Kern et al. (2023)
###
### Computed for EVERY model, not only GPT-4o-mini. The appendix figure shows a
### single model; the table reports all seven, because the ranges differ a great
### deal between them and a range quoted from one model is not a range across
### models.
###
### Outputs
###   data_work/outputs/ladder_{fleiss_within,cohen_across,cohen_human,icc}.csv
###   data_work/outputs/table_agreement_ladder.tex
###   plots/agreements_few_new_conditions_2x1.pdf/.png
################################################################################

BASEDIR <- "/Users/sseckman/Downloads/Annotation_Sensitivity_in_Large_Language_Models__ACL_ (1)"
source(file.path(BASEDIR, "data_work", "src", "00_utils.R"))
suppressPackageStartupMessages({ library(lme4); library(purrr); library(patchwork) })

say <- make_logger("01_agreement.log")

## Which model the appendix matrix figure shows. Everything else is all-model.
FIG_MODEL <- "GPT-4o-mini"

OUTCOMES <- c("OL", "HS")
OUTCOME_NAME <- c(OL = "Offensive language", HS = "Hate speech")

## Condition order inside a panel: the three task structures, each in the four
## prompt variants. C.OL and C.HS are the same "Separate" structure seen from
## the two outcomes, so each panel has twelve columns.
VARIANTS   <- c("", "_conf", "_batch", "_batch_conf")
VAR_LABEL  <- c("base", "conf", "batch", "batch+conf")
STRUCTS    <- list(OL = c("A", "B", "C.OL"), HS = c("A", "B", "C.HS"))
STRUCT_LAB <- c(A = "Joint, OL first", B = "Joint, HS first",
                C.OL = "Separate", C.HS = "Separate")

cond_order <- function(outcome) {
  as.vector(outer(VARIANTS, STRUCTS[[outcome]], function(v, s) paste0(s, v)))
}

### ------------------------------------------------------------------- load

say("Loading run-level labels ...")
long <- load_long(cols = c("tweet_id", "condition", "model", "task_structure",
                           "responder", "OL", "HS"))
say("  df_long rows for the seven models: %s", format(nrow(long), big.mark = ","))

say("Loading modal labels ...")
agg <- load_agg()
say("  df_agg rows for the seven models: %s", format(nrow(agg), big.mark = ","))

hum <- human_majority()
say("  human reference tweets: %d", nrow(hum))

### -------------------------------------------- rung 1: Fleiss within a design

say("\n%s\nRUNG 1  WITHIN A DESIGN (Fleiss' kappa over three runs)\n%s",
    strrep("=", 78), strrep("=", 78))

fleiss_tab <- cached("fleiss_within", map_dfr(OUTCOMES, function(oc) {
  e <- eligible(long, oc)
  e %>%
    group_by(model, condition) %>%
    group_modify(function(d, key) {
      cnt <- d %>% group_by(tweet_id) %>%
        summarise(n_pos = sum(.data[[oc]]), n = dplyr::n(), .groups = "drop") %>%
        filter(n == 3)
      data.frame(kappa = fleiss_kappa_binary(cnt$n_pos, 3), n_items = nrow(cnt))
    }) %>%
    ungroup() %>%
    mutate(outcome = oc)
}))
write.csv(fleiss_tab, file.path(OUT, "ladder_fleiss_within.csv"), row.names = FALSE)

### ------------------------------------------ rung 2: Cohen across two designs

say("\n%s\nRUNG 2  ACROSS DESIGNS (Cohen's kappa between modal labels)\n%s",
    strrep("=", 78), strrep("=", 78))

cohen_tab <- cached("cohen_across", map_dfr(OUTCOMES, function(oc) {
  e <- eligible(agg, oc)
  mcol <- paste0("modal_", oc)
  map_dfr(MODEL_ORDER, function(m) {
    w <- e %>% filter(model == m) %>%
      select(tweet_id, condition, value = all_of(mcol)) %>%
      pivot_wider(names_from = condition, values_from = value)
    cs <- setdiff(names(w), "tweet_id")
    pairs <- t(combn(cs, 2))
    data.frame(model = m, outcome = oc, cond_a = pairs[, 1], cond_b = pairs[, 2],
               kappa = apply(pairs, 1, function(p) cohens_kappa(w[[p[1]]], w[[p[2]]])))
  })
}))
write.csv(cohen_tab, file.path(OUT, "ladder_cohen_across.csv"), row.names = FALSE)

### ------------------------------------------- rung 3: Cohen against the humans

say("\n%s\nRUNG 3  AGAINST THE HUMAN MAJORITY (Cohen's kappa)\n%s",
    strrep("=", 78), strrep("=", 78))

human_tab <- cached("cohen_human", map_dfr(OUTCOMES, function(oc) {
  e <- eligible(agg, oc) %>%
    left_join(hum[, c("tweet_id", paste0("human_", oc, "_maj"))], by = "tweet_id")
  e %>% group_by(model, condition) %>%
    summarise(kappa = cohens_kappa(.data[[paste0("modal_", oc)]],
                                   .data[[paste0("human_", oc, "_maj")]]),
              .groups = "drop") %>%
    mutate(outcome = oc)
}))
write.csv(human_tab, file.path(OUT, "ladder_cohen_human.csv"), row.names = FALSE)

### --------------------------------------------------- tweet-level ICC per model

say("\n%s\nTWEET-LEVEL ICC (per-model LPM with design factors)\n%s",
    strrep("=", 78), strrep("=", 78))

icc_tab <- cached("icc_per_model", map_dfr(OUTCOMES, function(oc) {
  e <- eligible(long, oc) %>%
    mutate(joint_OL_first = as.integer(task_structure == "A"),
           joint_HS_first = as.integer(task_structure == "B"),
           conf_cond      = as.integer(grepl("conf", condition)),
           batched_cond   = as.integer(grepl("batch", condition)))
  map_dfr(MODEL_ORDER, function(m) {
    d <- e[e$model == m, ]
    f <- lmer(as.formula(paste0(oc, " ~ joint_OL_first + joint_HS_first +",
                                " conf_cond + batched_cond + (1 | tweet_id)")),
              data = d, REML = TRUE,
              control = lmerControl(calc.derivs = FALSE))
    v <- as.data.frame(VarCorr(f))
    vt <- v$vcov[v$grp == "tweet_id"]; vr <- v$vcov[v$grp == "Residual"]
    data.frame(model = m, outcome = oc, icc = vt / (vt + vr))
  })
}))
write.csv(icc_tab, file.path(OUT, "ladder_icc.csv"), row.names = FALSE)

for (m in MODEL_ORDER) {
  say("  %-20s OL ICC = %.3f   HS ICC = %.3f", m,
      icc_tab$icc[icc_tab$model == m & icc_tab$outcome == "OL"],
      icc_tab$icc[icc_tab$model == m & icc_tab$outcome == "HS"])
}
pooled_icc <- cached("icc_pooled", map_dfr(OUTCOMES, function(oc) {
  e <- eligible(long, oc) %>%
    mutate(joint_OL_first = as.integer(task_structure == "A"),
           joint_HS_first = as.integer(task_structure == "B"),
           conf_cond      = as.integer(grepl("conf", condition)),
           batched_cond   = as.integer(grepl("batch", condition)))
  f <- lmer(as.formula(paste0(oc, " ~ joint_OL_first + joint_HS_first +",
                              " conf_cond + batched_cond + (1 | tweet_id)")),
            data = e, REML = TRUE, control = lmerControl(calc.derivs = FALSE))
  v <- as.data.frame(VarCorr(f))
  data.frame(outcome = oc,
             icc = v$vcov[v$grp == "tweet_id"] /
                   (v$vcov[v$grp == "tweet_id"] + v$vcov[v$grp == "Residual"]))
}))
say("  pooled over models: OL %.3f   HS %.3f",
    pooled_icc$icc[pooled_icc$outcome == "OL"],
    pooled_icc$icc[pooled_icc$outcome == "HS"])

### ------------------------------------------------------ the ladder, per model

say("\n%s\nTHE LADDER, PER MODEL (pooled over both outcomes)\n%s",
    strrep("=", 78), strrep("=", 78))
say("%-20s %-18s %-18s %-18s", "model", "within (Fleiss)", "across (Cohen)",
    "vs human (Cohen)")

rng <- function(x) { x <- x[is.finite(x)]
  sprintf("%.2f-%.2f (%.2f)", min(x), max(x), mean(x)) }
fin <- function(x) x[is.finite(x)]
ladder <- map_dfr(MODEL_ORDER, function(m) {
  w <- fin(fleiss_tab$kappa[fleiss_tab$model == m])
  a <- fin(cohen_tab$kappa[cohen_tab$model == m])
  h <- fin(human_tab$kappa[human_tab$model == m])
  say("%-20s %-18s %-18s %-18s", m, rng(w), rng(a), rng(h))
  data.frame(model = m,
             within_min = min(w), within_max = max(w), within_mean = mean(w),
             across_min = min(a), across_max = max(a), across_mean = mean(a),
             human_min = min(h),  human_max = max(h),  human_mean = mean(h))
})
write.csv(ladder, file.path(OUT, "ladder_per_model.csv"), row.names = FALSE)

## A handful of model x condition cells are degenerate for Fleiss: the model
## gave the same label to every tweet, so chance agreement is 1 and kappa is
## undefined. Those cells are dropped, and counted, rather than treated as 0.
n_degen <- sum(!is.finite(fleiss_tab$kappa))
if (n_degen > 0) {
  say("\n%d of %d model x condition cells are degenerate for Fleiss' kappa",
      n_degen, nrow(fleiss_tab))
  print(fleiss_tab[!is.finite(fleiss_tab$kappa), c("model", "condition", "outcome")])
}
fleiss_tab <- fleiss_tab[is.finite(fleiss_tab$kappa) | TRUE, ]
allw <- fleiss_tab$kappa[is.finite(fleiss_tab$kappa)]
alla <- cohen_tab$kappa[is.finite(cohen_tab$kappa)]
allh <- human_tab$kappa[is.finite(human_tab$kappa)]
say("\nPooled over all seven models:")
say("  within a design    %.3f - %.3f   median %.3f", min(allw), max(allw), median(allw))
say("  across designs     %.3f - %.3f   median %.3f", min(alla), max(alla), median(alla))
say("  vs human majority  %.3f - %.3f   median %.3f", min(allh), max(allh), median(allh))

six <- setdiff(MODEL_ORDER, "Llama-3.1-8B")
sw <- fin(fleiss_tab$kappa[fleiss_tab$model %in% six])
sa <- fin(cohen_tab$kappa[cohen_tab$model %in% six])
sh <- fin(human_tab$kappa[human_tab$model %in% six])
say("Excluding Llama 3.1 8B:")
say("  within a design    %.3f - %.3f   median %.3f", min(sw), max(sw), median(sw))
say("  across designs     %.3f - %.3f   median %.3f", min(sa), max(sa), median(sa))
say("  vs human majority  %.3f - %.3f   median %.3f", min(sh), max(sh), median(sh))

say("\nOrdering check (does the ladder hold model by model?):")
ord_mean <- with(ladder, within_mean > across_mean & across_mean > human_mean)
say("  within > across > human on the MEANS: %d of %d models", sum(ord_mean), nrow(ladder))
ord_rng <- with(ladder, within_min > across_max)
say("  min within > max across (range separation): %d of %d models",
    sum(ord_rng), nrow(ladder))

### -------------------------------------------------------------- LaTeX table

## kappa formatting: strip the leading zero, render negatives with a LaTeX minus
k2 <- function(x) {
  s   <- sprintf("%.2f", x)
  neg <- grepl("^-", s)
  s   <- sub("^0", "", sub("^-", "", s))
  ifelse(neg, paste0("$-$", s), s)
}
fmt <- function(lo, hi, mu) sprintf("%s--%s (%s)", k2(lo), k2(hi), k2(mu))
tex <- c("% auto-generated by data_work/src/01_agreement.R",
         "\\begin{tabular}{@{}lccc@{}}", "\\toprule",
         "Model & Within a design & Across designs & Against humans \\\\",
         " & (Fleiss' $\\kappa$) & (Cohen's $\\kappa$) & (Cohen's $\\kappa$) \\\\",
         "\\midrule")
for (i in seq_len(nrow(ladder))) {
  r <- ladder[i, ]
  tex <- c(tex, sprintf("%s & %s & %s & %s \\\\", MODEL_LABELS[[r$model]],
                        fmt(r$within_min, r$within_max, r$within_mean),
                        fmt(r$across_min, r$across_max, r$across_mean),
                        fmt(r$human_min, r$human_max, r$human_mean)))
}
tex <- c(tex, "\\midrule",
         sprintf("All models & %s & %s & %s \\\\",
                 fmt(min(allw), max(allw), median(allw)),
                 fmt(min(alla), max(alla), median(alla)),
                 fmt(min(allh), max(allh), median(allh))),
         sprintf("Excluding Llama 3.1 8B & %s & %s & %s \\\\",
                 fmt(min(sw), max(sw), median(sw)),
                 fmt(min(sa), max(sa), median(sa)),
                 fmt(min(sh), max(sh), median(sh))),
         "\\bottomrule", "\\end{tabular}")
writeLines(tex, file.path(OUT, "table_agreement_ladder.tex"))
say("\nWrote %s/table_agreement_ladder.tex", OUT)

### ----------------------------------------------------------- matrix figure

## Lower triangle: Fleiss on the diagonal, Cohen off it, plus a reference column.
matrix_panel_data <- function(m, oc) {
  cs <- cond_order(oc)
  idx <- setNames(seq_along(cs), cs)
  d <- cohen_tab %>% filter(model == m, outcome == oc) %>%
    mutate(i = idx[cond_a], j = idx[cond_b]) %>%
    transmute(row = pmax(i, j), col = pmin(i, j), kappa)
  diag <- fleiss_tab %>% filter(model == m, outcome == oc) %>%
    mutate(row = idx[condition], col = row) %>% select(row, col, kappa)
  ref <- human_tab %>% filter(model == m, outcome == oc) %>%
    mutate(row = idx[condition], col = length(cs) + 1L) %>% select(row, col, kappa)
  bind_rows(diag, d, ref) %>% filter(!is.na(row)) %>%
    mutate(outcome = factor(OUTCOME_NAME[[oc]], levels = OUTCOME_NAME),
           lab = sub("^0", "", sprintf("%.2f", kappa)))
}

NCOL <- 13L
grp_at <- c(2.5, 6.5, 10.5)
grp_nm <- c("Joint, OL first", "Joint, HS first", "Separate")

## Both axes carry a two-level label: the task structure spanning four rows, and
## the prompt variant inside it. ggplot has no nested discrete axis, so both
## levels are drawn as text outside the panel with clipping off. Row labels are
## right-anchored and structure labels left-anchored, so they grow away from each
## other and cannot collide. The two outcomes are separate plots stacked with
## patchwork rather than facets, so that the cells stay the same size even though
## only the lower panel reserves space for the column labels.
panel_plot <- function(oc, with_x) {
  d <- matrix_panel_data(FIG_MODEL, oc)
  ymax <- if (with_x) 15.9 else 12.6
  row_lab <- data.frame(y = 1:12, lab = rep(VAR_LABEL, 3), x = 0.35)
  grp_lab <- data.frame(y = grp_at, lab = grp_nm, x = -3.5)
  g <- ggplot(d, aes(x = col, y = row, fill = kappa)) +
    geom_tile(colour = "white", linewidth = 0.35) +
    geom_text(aes(label = lab, colour = kappa > 0.86), size = 1.45,
              show.legend = FALSE) +
    scale_colour_manual(values = c(`TRUE` = "white", `FALSE` = "black"),
                        guide = "none") +
    scale_fill_distiller(palette = "RdBu", direction = -1, limits = c(0, 1),
                         breaks = c(0, .25, .5, .75, 1),
                         name = "Kappa (Cohen's or Fleiss')") +
    scale_x_continuous(limits = c(-3.6, NCOL + 0.6), expand = expansion(0, 0)) +
    scale_y_reverse(limits = c(ymax, 0.4), expand = expansion(0, 0)) +
    geom_text(data = row_lab, aes(x, y, label = lab), inherit.aes = FALSE,
              size = 1.5, hjust = 1, colour = "grey20") +
    geom_text(data = grp_lab, aes(x, y, label = lab), inherit.aes = FALSE,
              size = 1.6, hjust = 0) +
    labs(title = OUTCOME_NAME[[oc]]) +
    coord_cartesian(clip = "off") +
    theme_paper(6) +
    theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
          axis.line = element_blank(), axis.ticks = element_blank(),
          axis.title = element_blank(), axis.text = element_blank(),
          plot.title = element_text(size = 6, face = "bold", hjust = 0.55),
          legend.position = "bottom", legend.direction = "horizontal",
          legend.title = element_text(size = 5.4),
          legend.text = element_text(size = 4.6),
          legend.key.height = unit(4, "pt"), legend.key.width = unit(26, "pt"),
          plot.margin = margin(2, 2, 2, 2)) +
    guides(fill = guide_colourbar(title.position = "top", title.hjust = 0.5,
                                 ticks.colour = "white"))
  if (with_x) {
    col_lab <- data.frame(x = 1:NCOL, y = 12.7,
                          lab = c(rep(VAR_LABEL, 3), "Reference"))
    colgrp <- data.frame(x = grp_at, y = 15.5, lab = grp_nm)
    g <- g +
      geom_text(data = col_lab, aes(x, y, label = lab), inherit.aes = FALSE,
                size = 1.5, hjust = 1, angle = 90, colour = "grey20") +
      geom_text(data = colgrp, aes(x, y, label = lab), inherit.aes = FALSE,
                size = 1.6, hjust = 0.5)
  }
  g
}

p <- (panel_plot("OL", FALSE) / panel_plot("HS", TRUE)) +
  patchwork::plot_layout(heights = c(12.6, 15.9), guides = "collect") &
  theme(legend.position = "bottom")

save_fig(p, "agreements_few_new_conditions_2x1", width = 3.35, height = 4.5)

say("\nMatrix figure shows %s only. Its within-design range is %.3f-%.3f,",
    FIG_MODEL, min(fleiss_tab$kappa[fleiss_tab$model == FIG_MODEL]),
    max(fleiss_tab$kappa[fleiss_tab$model == FIG_MODEL]))
say("which is NOT the range across models (%.3f-%.3f). Caption must say so.",
    min(allw), max(allw))
say("\nDone.")
