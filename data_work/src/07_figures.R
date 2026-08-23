################################################################################
### 07_figures.R  --  fig:design_by_model  (plots/design_effects_families.pdf)
###
### The sensitivity claim in one picture: for each of the four design factors,
### where does each model put the effect, and how far apart are the seven
### answers? Colour encodes the model family, shape the position within the
### family, and the black bar is the pooled mixed-effects estimate that a single
### aggregate regression would report.
###
### Reads (both written by 02_regressions.R, so run that first):
###   data_work/outputs/permodel_coefs.csv
###   data_work/outputs/pooled_coefs.csv
################################################################################

BASEDIR <- "/Users/sseckman/Downloads/Annotation_Sensitivity_in_Large_Language_Models__ACL_ (1)"
source(file.path(BASEDIR, "data_work", "src", "00_utils.R"))

say <- make_logger("07_figures.log")

OUTCOMES     <- c("OL", "HS")
OUTCOME_NAME <- c(OL = "Offensive language", HS = "Hate speech")

## Top to bottom, as in the submitted figure.
TERMS <- c(joint_OL_first = "Joint, OL first",
           joint_HS_first = "Joint, HS first",
           confidence     = "With confidence",
           batched        = "Batch prompt")

## Vertical offset by family so that near-identical estimates stay legible.
FAM_DY   <- c(OpenAI = 0.20, Mistral = 0.0, Meta = -0.20)
POOL_LAB <- "pooled estimate"
BAR_HALF <- 0.30           # half-height of the pooled bar, in row units

### ---------------------------------------------------------------------- data

need <- file.path(OUT, c("permodel_coefs.csv", "pooled_coefs.csv"))
if (!all(file.exists(need)))
  stop("run 02_regressions.R first; missing: ",
       paste(basename(need[!file.exists(need)]), collapse = ", "))

permodel <- readr::read_csv(need[1], show_col_types = FALSE)
pooled   <- readr::read_csv(need[2], show_col_types = FALSE)

## Row position: first term at the top.
yof <- function(term) length(TERMS) + 1 - match(term, names(TERMS))

pts <- permodel %>%
  filter(term %in% names(TERMS)) %>%
  mutate(x       = est * 100,
         fam     = MODEL_FAM[model],
         y       = yof(term) + FAM_DY[fam],
         series  = factor(MODEL_LABELS[model], levels = MODEL_LABELS[MODEL_ORDER]),
         outcome = factor(outcome, levels = OUTCOMES))

bars <- pooled %>%
  filter(term %in% names(TERMS)) %>%
  mutate(x       = est * 100,
         y       = yof(term),
         outcome = factor(outcome, levels = OUTCOMES))

## The quantity the caption quotes: the spread of the seven per-model estimates.
sds <- permodel %>%
  filter(term %in% names(TERMS)) %>%
  group_by(outcome, term) %>%
  summarise(sd = sd(est) * 100, .groups = "drop") %>%
  mutate(y = yof(term), outcome = factor(outcome, levels = OUTCOMES),
         lab = sprintf("SD %.1f pp", sd))

say("Cross-model SD of each design effect (pp):")
for (oc in OUTCOMES) for (tm in names(TERMS))
  say("  %s  %-16s %.1f", oc, TERMS[[tm]], sds$sd[sds$outcome == oc & sds$term == tm])

say("\nPooled estimates (pp):")
for (oc in OUTCOMES) for (tm in names(TERMS))
  say("  %s  %-16s %+.2f", oc, TERMS[[tm]],
      bars$x[bars$outcome == oc & bars$term == tm])

## The three claims the surrounding paragraph makes about this figure.
say("\nDirectional consistency (how many of the seven models share the sign,")
say("and the largest exception in the other direction):")
for (oc in OUTCOMES) for (tm in names(TERMS)) {
  e   <- permodel$est[permodel$outcome == oc & permodel$term == tm] * 100
  n   <- max(sum(e > 0), sum(e < 0))
  opp <- if (sum(e > 0) >= sum(e < 0)) e[e < 0] else e[e > 0]
  say("  %s  %-16s %d of 7 %s   worst exception %s pp", oc, TERMS[[tm]], n,
      if (sum(e > 0) >= sum(e < 0)) "raise " else "lower ",
      if (length(opp)) sprintf("%+.2f", opp[which.max(abs(opp))]) else "none")
}

say("\nMean cross-model SD over the four factors:")
for (oc in OUTCOMES) say("  %s  %.1f pp", oc, mean(sds$sd[sds$outcome == oc]))

## Dropping the cheapest model: the paper quotes batch-OL and confidence-HS.
say("\nCross-model SD excluding Llama-3.1-8B (pp):")
sd_ex <- permodel %>%
  filter(term %in% names(TERMS), model != "Llama-3.1-8B") %>%
  group_by(outcome, term) %>% summarise(sd = sd(est) * 100, .groups = "drop")
for (oc in OUTCOMES) for (tm in names(TERMS))
  say("  %s  %-16s %.1f  (was %.1f)", oc, TERMS[[tm]],
      sd_ex$sd[sd_ex$outcome == oc & sd_ex$term == tm],
      sds$sd[sds$outcome == oc & sds$term == tm])

### -------------------------------------------------------------------- figure

## One shape/colour scale covers the seven models and the pooled bar. The bar's
## key is the "|" glyph (pch 124), which is why the pooled estimate can sit in
## the same legend block as the models instead of needing a second legend.
SERIES     <- c(MODEL_LABELS[MODEL_ORDER], POOL_LAB)
SER_COLOUR <- c(FAM_COLORS[MODEL_FAM[MODEL_ORDER]], "black")
SER_SHAPE  <- c(MODEL_SHAPE[MODEL_ORDER], 124)
names(SER_COLOUR) <- names(SER_SHAPE) <- SERIES

bars$series <- factor(POOL_LAB, levels = SERIES)

xr  <- range(c(pts$x, bars$x))
pad <- diff(xr) * 0.045
SD_X <- xr[2] + diff(xr) * 0.13

p <- ggplot() +
  geom_vline(xintercept = 0, colour = "grey60", linewidth = 0.28,
             linetype = "dashed") +
  ## The bar first, so the "|" key glyph lands on top of it.
  geom_linerange(data = bars,
                 aes(x = x, ymin = y - BAR_HALF, ymax = y + BAR_HALF),
                 colour = "black", linewidth = 0.85) +
  geom_point(data = pts, aes(x, y, colour = series, shape = series),
             size = 1.35, stroke = 0.5, fill = NA) +
  ## Drawn only so the pooled bar earns a key in the models' legend; the glyph
  ## lands on top of the linerange above and is invisible in the panel.
  geom_point(data = bars, aes(x, y, colour = series, shape = series),
             size = 2.8, stroke = 1.4) +
  geom_hline(yintercept = 0.455, colour = "black", linewidth = 0.3) +
  geom_text(data = sds, aes(x = SD_X, y = y, label = lab),
            size = 2.1, colour = "grey35", hjust = 0) +
  facet_wrap(~ outcome, ncol = 1, labeller = labeller(outcome = OUTCOME_NAME)) +
  scale_colour_manual(values = SER_COLOUR, breaks = SERIES, name = NULL) +
  scale_shape_manual(values = SER_SHAPE, breaks = SERIES, name = NULL) +
  scale_x_continuous(breaks = seq(-20, 10, 5)) +
  scale_y_continuous(breaks = seq_along(TERMS),
                     labels = rev(unname(TERMS)),
                     limits = c(0.45, length(TERMS) + 0.55), expand = c(0, 0)) +
  coord_cartesian(xlim = c(xr[1] - pad, xr[2] + pad), clip = "off") +
  labs(x = "Change in labeling rate (percentage points)", y = NULL) +
  theme_paper(7.0) +
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        axis.line.x = element_blank(),
        axis.line.y = element_blank(),
        axis.ticks.y = element_blank(),
        axis.text.y = element_text(size = 6.6, colour = "black"),
        strip.text = element_text(size = 7.4, face = "plain", hjust = 0,
                                  margin = margin(0, 0, 2, 0)),
        panel.spacing.y = unit(9, "pt"),
        plot.margin = margin(2, 46, 2, 2),
        legend.position = "bottom",
        legend.text = element_text(size = 6.2),
        legend.key.height = unit(7.5, "pt"),
        legend.spacing.x = unit(1, "pt")) +
  guides(colour = guide_legend(ncol = 2, byrow = FALSE),
         shape  = guide_legend(ncol = 2, byrow = FALSE))

save_fig(p, "design_effects_families", width = 3.35, height = 3.75)

say("\nDone.")
