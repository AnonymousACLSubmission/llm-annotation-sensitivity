################################################################################
### 04_human_design.R  --  human sensitivity on the same 3,000 tweets
###
### Kern et al. (2023) had 917 crowd annotators rate these same tweets under five
### questionnaire versions, three annotators per tweet per version, no annotator
### in two versions: 44,900 ratings. That is the human analogue of our design,
### three replicate label sets per design and designs that differ only in
### presentation.
###
### Kern's version letters and our task_structure letters are unrelated schemes.
### Below, "version A--E" is always Kern's; ours are always named descriptively.
###
###   version A  both items on one screen, hate speech first     joint
###   version B  two screens, hate speech screen first           separate, HS first
###   version C  two screens, offensive language first           separate  <- reference
###   version D  50 tweets of HS then 50 of OL                    separate, HS first, blocked
###   version E  50 tweets of OL then 50 of HS                    separate, blocked
###
### Three factors are estimable, and each has an LLM counterpart:
###   joint     both constructs in one call      task_structure A or B vs separate
###   hs_first  hate speech listed first          task_structure B vs A
###   blocked   items arrive in blocks            six tweets per call vs one
###
### Kern collected no confidence scores, so that factor has no human analogue.
###
### Output: tab:human_llm, plus the validation of Kern's published Table 1 and
### Table 2 that licenses using his data at all.
################################################################################

BASEDIR <- "/Users/sseckman/Downloads/Annotation_Sensitivity_in_Large_Language_Models__ACL_ (1)"
source(file.path(BASEDIR, "data_work", "src", "00_utils.R"))
suppressPackageStartupMessages({
  library(purrr); library(sandwich); library(lmtest); library(tidyr); library(lme4)
})

say <- make_logger("04_human_design.log")

set.seed(20260805)
N_ITEMS  <- 3000
N_PERM   <- 2000    # permutations for the replicate-panel SD
N_BOOT   <- 5000    # cluster-bootstrap reps for the annotator-panel SD
VERSIONS <- c("A", "B", "C", "D", "E")
OUTCOMES <- c(OL = "offensive_language", HS = "hate_speech")
FACTORS  <- c("joint", "hs_first", "blocked")

## Kern's version to design-feature map, read off his Section 3 and Figure 1.
KERN_DESIGN <- data.frame(
  version  = VERSIONS,
  joint    = c(1, 0, 0, 0, 0),
  hs_first = c(1, 1, 0, 1, 0),
  blocked  = c(0, 0, 0, 1, 1))

### --------------------------------------------------------------- human ratings

k <- load_kern() %>%
  filter(version %in% VERSIONS) %>%
  left_join(KERN_DESIGN, by = "version")

say("Kern ratings: %s   tweets: %s   annotators: %s",
    format(nrow(k), big.mark = ","),
    format(dplyr::n_distinct(k$tweet_id), big.mark = ","),
    format(dplyr::n_distinct(k$id), big.mark = ","))

### -------------------------------- 1. validation against Kern's Table 1 and 2

say("\n%s\n1. PREVALENCE BY VERSION  (validates Kern et al. Table 1)\n%s",
    strrep("=", 74), strrep("=", 74))

ver_prev <- map_dfr(VERSIONS, function(v) {
  d <- k[k$version == v, ]
  tibble::tibble(version = v, n = nrow(d),
                 OL = mean(d$offensive_language, na.rm = TRUE),
                 HS = mean(d$hate_speech, na.rm = TRUE))
})
say("%-5s %6s %8s %7s %8s %8s %8s", "ver", "joint", "HS 1st", "block", "n",
    "OL %", "HS %")
for (i in seq_len(nrow(ver_prev))) {
  f <- KERN_DESIGN[KERN_DESIGN$version == ver_prev$version[i], ]
  say("%-5s %6d %8d %7d %8s %8.1f %8.1f", ver_prev$version[i], f$joint,
      f$hs_first, f$blocked, format(ver_prev$n[i], big.mark = ","),
      ver_prev$OL[i] * 100, ver_prev$HS[i] * 100)
}
for (oc in names(OUTCOMES)) {
  v <- ver_prev[[oc]] * 100
  say("%s: range across versions %.1f - %.1f = %.1f pp,  SD %.2f pp", oc,
      min(v), max(v), max(v) - min(v), sd(v))
}

## Cross-version agreement on each version's majority label. Kern's Table 2
## reports Krippendorff's alpha within version (OL .629-.740, HS .477-.596);
## reproducing it is the check that our extract is his data.
k <- k %>% group_by(version, tweet_id) %>% mutate(slot = row_number()) %>% ungroup()

say("\n%s\n2. AGREEMENT WITHIN AND ACROSS VERSIONS  (validates Kern Table 2)\n%s",
    strrep("=", 74), strrep("=", 74))

maj <- k %>% group_by(version, tweet_id) %>%
  summarise(across(all_of(unname(OUTCOMES)), ~ mean(.x, na.rm = TRUE)),
            .groups = "drop")

flip_share <- c()
for (oc in names(OUTCOMES)) {
  col <- OUTCOMES[[oc]]
  say("\n--- %s ---", oc)
  say("  within a version, across the three annotators:")
  for (v in VERSIONS) {
    d <- k[k$version == v, ]
    g <- d %>% group_by(tweet_id) %>%
      summarise(s = sum(.data[[col]], na.rm = TRUE), n = sum(!is.na(.data[[col]])),
                .groups = "drop") %>% filter(n == 3)
    M <- d %>% select(tweet_id, slot, all_of(col)) %>%
      pivot_wider(names_from = slot, values_from = all_of(col)) %>%
      select(-tweet_id) %>% as.matrix()
    say("    %s: Fleiss kappa %.3f   Krippendorff alpha %.3f   (n = %s tweets)",
        v, fleiss_kappa_binary(g$s, 3), krippendorff_alpha_binary(M),
        format(nrow(g), big.mark = ","))
  }

  W <- maj %>% mutate(m = as.integer(.data[[col]] >= 0.5)) %>%
    select(version, tweet_id, m) %>%
    pivot_wider(names_from = version, values_from = m)
  say("  across versions, pairwise Cohen kappa on version majority labels:")
  ks <- c()
  for (i in 1:4) for (j in (i + 1):5) {
    kk <- cohens_kappa(W[[VERSIONS[i]]], W[[VERSIONS[j]]])
    ks <- c(ks, kk)
    say("    %s-%s: %.3f", VERSIONS[i], VERSIONS[j], kk)
  }
  say("  %s: range %.3f - %.3f, mean %.3f", oc, min(ks), max(ks), mean(ks))
  fl <- mean(apply(as.matrix(W[VERSIONS]), 1, function(r) sd(r) > 0))
  flip_share[oc] <- fl
  say("  share of tweets whose majority label differs across versions: %.1f%%",
      fl * 100)
}

### ------------------------------------------- 3. replicate panels (human "run")

say("\n%s\n3. REPLICATE VARIATION WITHIN A DESIGN\n%s", strrep("=", 74),
    strrep("=", 74))
say("Three annotators rate every tweet in every version. Randomly assigning them")
say("to three panels gives three complete label sets per version, the direct")
say("analogue of our three runs per condition. %d random assignments per version.",
    N_PERM)

rep_sd <- cached("human_replicate_sd", {
  out <- map_dfr(VERSIONS, function(v) {
    d <- k[k$version == v, ]
    res <- vapply(names(OUTCOMES), function(oc) {
      col <- OUTCOMES[[oc]]
      y   <- d[[col]]
      tw  <- d$tweet_id
      sds <- vapply(seq_len(N_PERM), function(b) {
        ## Shuffle within tweet, then take the first, second and third
        ## annotator of each tweet as three complete label sets.
        o  <- order(tw, runif(length(tw)))
        s  <- ave(seq_along(o), tw[o], FUN = seq_along)
        p  <- tapply(y[o], s, mean, na.rm = TRUE)[1:3]
        sd(p)
      }, numeric(1))
      mean(sds)
    }, numeric(1))
    tibble::tibble(version = v, OL = res[["OL"]], HS = res[["HS"]])
  })
  out
})
for (i in seq_len(nrow(rep_sd)))
  say("  %s: replicate SD  OL %.2f pp   HS %.2f pp", rep_sd$version[i],
      rep_sd$OL[i] * 100, rep_sd$HS[i] * 100)

## Components are combined across versions as a root mean square, because they
## are variances of the same quantity measured five times, not five estimates to
## be averaged on the SD scale.
rms <- function(x) sqrt(mean(x^2))
human_rep    <- vapply(names(OUTCOMES), function(oc) rms(rep_sd[[oc]]), numeric(1))
human_design <- vapply(names(OUTCOMES), function(oc) sd(ver_prev[[oc]]), numeric(1))
for (oc in names(OUTCOMES))
  say("%s: replicate SD (RMS over versions) %.2f pp | design SD across versions %.2f pp | ratio %.1fx",
      oc, human_rep[[oc]] * 100, human_design[[oc]] * 100,
      human_design[[oc]] / human_rep[[oc]])

### -------------------------------------------- 4. annotator-panel sampling

say("\n%s\n4. ANNOTATOR-PANEL SAMPLING\n%s", strrep("=", 74), strrep("=", 74))
say("Resampling annotators within a version (cluster bootstrap, %d reps) gives the",
    N_BOOT)
say("uncertainty a study incurs by hiring one panel of ~180 crowd workers rather")
say("than another. LLMs have no counterpart to this term.")

panel_sd <- cached("human_panel_sd", {
  map_dfr(VERSIONS, function(v) {
    d <- k[k$version == v, ]
    ids <- unique(d$id)
    res <- vapply(names(OUTCOMES), function(oc) {
      col <- OUTCOMES[[oc]]
      tab <- d %>% group_by(id) %>%
        summarise(s = sum(.data[[col]], na.rm = TRUE),
                  n = sum(!is.na(.data[[col]])), .groups = "drop")
      tab <- tab[match(ids, tab$id), ]
      draws <- matrix(sample.int(length(ids), N_BOOT * length(ids), replace = TRUE),
                      nrow = N_BOOT)
      ## Ratio-of-sums estimator: the prevalence a resampled panel would report.
      p <- rowSums(matrix(tab$s[draws], nrow = N_BOOT)) /
           rowSums(matrix(tab$n[draws], nrow = N_BOOT))
      sd(p)
    }, numeric(1))
    tibble::tibble(version = v, OL = res[["OL"]], HS = res[["HS"]])
  })
})
for (i in seq_len(nrow(panel_sd)))
  say("  %s: annotator-panel SD  OL %.2f pp   HS %.2f pp", panel_sd$version[i],
      panel_sd$OL[i] * 100, panel_sd$HS[i] * 100)
human_panel <- vapply(names(OUTCOMES), function(oc) rms(panel_sd[[oc]]), numeric(1))
for (oc in names(OUTCOMES))
  say("  %s: annotator-panel SD (RMS over versions) %.2f pp", oc,
      human_panel[[oc]] * 100)

### ------------------------------------- 5. human design effects, clustered SEs

say("\n%s\n5. HUMAN LINEAR PROBABILITY MODELS (reference = version C)\n%s",
    strrep("=", 74), strrep("=", 74))
say("Annotator-clustered SEs are the headline; tweet-clustered are given for")
say("comparison, since each tweet is rated many times.")

human_coef <- list()
for (oc in names(OUTCOMES)) {
  col <- OUTCOMES[[oc]]
  dk  <- k[, c("id", "tweet_id", "version", FACTORS, col)] %>% tidyr::drop_na()
  fit <- lm(reformulate(FACTORS, response = col), data = dk)
  say("\n--- %s (n = %s ratings) ---", oc, format(nrow(dk), big.mark = ","))
  say("  intercept (version C) %.1f%%", coef(fit)[["(Intercept)"]] * 100)
  for (cl in c(id = "annotator", tweet_id = "tweet")) {
    gv <- if (cl == "annotator") dk$id else dk$tweet_id
    ct <- coeftest(fit, vcov = vcovCL(fit, cluster = gv, type = "HC1"))
    say("  %s-clustered SEs:", cl)
    for (t in FACTORS)
      say("    %-10s%+7.2f pp  (SE %.2f, p = %.4g)", t, ct[t, 1] * 100,
          ct[t, 2] * 100, ct[t, 4])
    if (cl == "annotator")
      human_coef[[oc]] <- data.frame(term = FACTORS, est = ct[FACTORS, 1],
                                     se = ct[FACTORS, 2], p = ct[FACTORS, 4])
  }
  ## Omnibus: do all five versions share one prevalence? Wald test on the
  ## version dummies with annotator-clustered covariance.
  dk$version <- relevel(factor(dk$version), ref = "C")
  fv <- lm(reformulate("version", response = col), data = dk)
  V  <- vcovCL(fv, cluster = dk$id, type = "HC1")
  w  <- waldtest(fv, vcov = V, test = "F")
  say("  omnibus test that all five versions share one prevalence: F = %.2f, p = %.2e",
      w$F[2], w$`Pr(>F)`[2])
  ct <- coeftest(fv, vcov = V)
  say("  version contrasts vs C (annotator-clustered):")
  for (t in setdiff(rownames(ct), "(Intercept)"))
    say("    %-10s%+7.2f pp  (SE %.2f, p = %.4g)", sub("^version", "", t),
        ct[t, 1] * 100, ct[t, 2] * 100, ct[t, 4])
}

### ------------------------------ 6. the same three factors on the LLM labels

say("\n%s\n6. THE SAME THREE FACTORS, ESTIMATED ON THE LLM LABELS\n%s",
    strrep("=", 74), strrep("=", 74))
say("joint    = both constructs in one call vs separate calls")
say("hs_first = hate speech listed first, within the joint conditions")
say("blocked  = six tweets per call vs one tweet per call")
say("Pooled OLS with model fixed effects, SEs clustered on tweet.")

long <- load_long(cols = c("tweet_id", "condition", "model", "task_structure",
                           "responder", "OL", "HS"))

llm_frames <- lapply(names(OUTCOMES), function(oc) {
  eligible(long, oc) %>%
    mutate(joint    = as.integer(task_structure %in% c("A", "B")),
           hs_first = as.integer(task_structure == "B"),
           blocked  = as.integer(grepl("batch", condition)),
           model    = factor(model, levels = MODEL_ORDER)) %>%
    filter(!is.na(.data[[oc]]))
})
names(llm_frames) <- names(OUTCOMES)

llm_coef <- list()
for (oc in names(OUTCOMES)) {
  e   <- llm_frames[[oc]]
  fit <- lm(reformulate(c(FACTORS, "model"), response = oc), data = e)
  ct  <- coeftest(fit, vcov = vcovCL(fit, cluster = e$tweet_id, type = "HC1"))
  say("\n--- %s (n = %s labels) ---", oc, format(nrow(e), big.mark = ","))
  for (t in FACTORS)
    say("  %-10s%+7.2f pp  (SE %.2f, p = %.3e)", t, ct[t, 1] * 100,
        ct[t, 2] * 100, ct[t, 4])
  llm_coef[[oc]] <- data.frame(term = FACTORS, est = ct[FACTORS, 1],
                               se = ct[FACTORS, 2], p = ct[FACTORS, 4])

  say("  per-model coefficients (pp):")
  say("    %-20s%9s%10s%9s", "model", "joint", "hs_first", "blocked")
  for (mo in MODEL_ORDER) {
    em <- e[e$model == mo, ]
    mm <- lm(reformulate(FACTORS, response = oc), data = em)
    b  <- coef(mm)
    say("    %-20s%+9.2f%+10.2f%+9.2f", mo, b[["joint"]] * 100,
        b[["hs_first"]] * 100, b[["blocked"]] * 100)
  }
}

say("\nHuman versus LLM design coefficients (pp), same three factors:")
say("  %-10s%10s%9s%11s%9s", "", "OL human", "OL LLM", "HS human", "HS LLM")
for (t in FACTORS)
  say("  %-10s%+10.2f%+9.2f%+11.2f%+9.2f", t,
      human_coef$OL$est[human_coef$OL$term == t] * 100,
      llm_coef$OL$est[llm_coef$OL$term == t] * 100,
      human_coef$HS$est[human_coef$HS$term == t] * 100,
      llm_coef$HS$est[llm_coef$HS$term == t] * 100)

### ------------------------------------- 7. one scale: tab:human_llm

say("\n%s\n7. HUMANS AND LLMs ON ONE SCALE  (tab:human_llm)\n%s",
    strrep("=", 74), strrep("=", 74))

## LLM components use the same crossed REML decomposition as 03_deff.R.
## Task-design variance combines the design and Model x Design components.
llm <- lapply(names(OUTCOMES), function(oc) {
  run <- eligible(long, oc) %>%
    group_by(model, condition, responder) %>%
    summarise(prev = mean(.data[[oc]], na.rm = TRUE), .groups = "drop")
  z <- crossed_prevalence_components(run, N_ITEMS)
  list(run = z$component_sd[["run"]],
       design = z$component_sd[["design"]],
       model = z$component_sd[["model"]],
       grand = z$prevalence,
       raw_design = sqrt(z$raw_var[["design"]]),
       model_design = sqrt(z$raw_var[["model_design"]]))
})
names(llm) <- names(OUTCOMES)

## Human total omits the replicate term: a study reports one panel's labels, and
## the panel-sampling term already covers who was hired. Including replicate as
## well would double-count the same annotators.
tabrows <- lapply(names(OUTCOMES), function(oc) {
  ph <- mean(ver_prev[[oc]]); pl <- llm[[oc]]$grand
  nom_h <- sqrt(ph * (1 - ph) / N_ITEMS)
  nom_l <- sqrt(pl * (1 - pl) / N_ITEMS)
  tot_h   <- sqrt(nom_h^2 + human_panel[[oc]]^2 + human_design[[oc]]^2)
  tot_l   <- sqrt(nom_l^2 + llm[[oc]]$run^2 + llm[[oc]]$design^2)
  tot_l_m <- sqrt(tot_l^2 + llm[[oc]]$model^2)
  list(oc = oc, ph = ph, pl = pl, nom_h = nom_h, nom_l = nom_l,
       rep_h = human_rep[[oc]], rep_l = llm[[oc]]$run,
       pan_h = human_panel[[oc]],
       des_h = human_design[[oc]], des_l = llm[[oc]]$design,
       mod_l = llm[[oc]]$model,
       deff_h = (tot_h / nom_h)^2, deff_l = (tot_l / nom_l)^2,
       deff_lm = (tot_l_m / nom_l)^2)
})
names(tabrows) <- names(OUTCOMES)

for (oc in names(OUTCOMES)) {
  r <- tabrows[[oc]]
  say("\n--- %s ---", oc)
  say("  %-34s%12s%12s", "", "humans", "LLMs")
  say("  %-34s%11.1f%%%11.1f%%", "prevalence", r$ph * 100, r$pl * 100)
  say("  %-34s%11.2f %11.2f", "replicate SD (panels / runs)", r$rep_h * 100,
      r$rep_l * 100)
  say("  %-34s%11.2f %11s", "annotator-panel SD", r$pan_h * 100, "--")
  say("  %-34s%11.2f %11.2f", "task-design SD", r$des_h * 100, r$des_l * 100)
  say("  %-34s%11s %11.2f", "model-choice SD", "--", r$mod_l * 100)
  say("  %-34s%11.2f %11.2f", "nominal SE at n = 3,000", r$nom_h * 100,
      r$nom_l * 100)
  say("  %-34s%11.1f %11.1f", "design effect", r$deff_h, r$deff_l)
  say("  %-34s%11s %11.1f", "  ... incl. model choice", "--", r$deff_lm)
  say("  %-34s%11.1f %11.1f", "design SD / replicate SD",
      r$des_h / r$rep_h, r$des_l / r$rep_l)
}

pm  <- function(x, d = 2) {
  s <- formatC(abs(x), format = "f", digits = d)
  if (x < 0) paste0("$-$", s) else paste0("$+$", s)
}
num <- function(x, d = 2) formatC(x, format = "f", digits = d)
O <- tabrows$OL; H <- tabrows$HS

lines <- c(
  "\\multicolumn{5}{@{}l}{\\emph{Variance components}} \\\\",
  sprintf("Prevalence & %.1f\\%% & %.1f\\%% & %.1f\\%% & %.1f\\%% \\\\",
          O$ph * 100, O$pl * 100, H$ph * 100, H$pl * 100),
  sprintf("\\quad Replicate SD & %s & %s & %s & %s \\\\",
          num(O$rep_h * 100), num(O$rep_l * 100), num(H$rep_h * 100), num(H$rep_l * 100)),
  sprintf("\\quad Annotator-panel SD & %s & --- & %s & --- \\\\",
          num(O$pan_h * 100), num(H$pan_h * 100)),
  sprintf("\\quad Task-design SD & %s & %s & %s & %s \\\\",
          num(O$des_h * 100), num(O$des_l * 100), num(H$des_h * 100), num(H$des_l * 100)),
  sprintf("\\quad Model-choice SD & --- & %s & --- & %s \\\\",
          num(O$mod_l * 100), num(H$mod_l * 100)),
  sprintf("Design effect & %.0f & %.0f & %.0f & %.0f \\\\",
          O$deff_h, O$deff_l, H$deff_h, H$deff_l),
  sprintf("\\quad incl.\\ model choice & --- & %.0f & --- & %.0f \\\\",
          O$deff_lm, H$deff_lm),
  "\\addlinespace[3pt]",
  "\\multicolumn{5}{@{}l}{\\emph{Effect of a shared design feature (pp)}} \\\\",
  sprintf("Both items together & %s & %s & %s & %s \\\\",
          pm(human_coef$OL$est[1] * 100), pm(llm_coef$OL$est[1] * 100),
          pm(human_coef$HS$est[1] * 100), pm(llm_coef$HS$est[1] * 100)),
  sprintf("Hate speech asked first & %s & %s & %s & %s \\\\",
          pm(human_coef$OL$est[2] * 100), pm(llm_coef$OL$est[2] * 100),
          pm(human_coef$HS$est[2] * 100), pm(llm_coef$HS$est[2] * 100)),
  sprintf("Items blocked / batched & %s & %s & %s & %s \\\\",
          pm(human_coef$OL$est[3] * 100), pm(llm_coef$OL$est[3] * 100),
          pm(human_coef$HS$est[3] * 100), pm(llm_coef$HS$est[3] * 100)))

write_tabular(lines, "table_human_llm.tex", "@{}lrrrr@{}", header = c(
  " & \\multicolumn{2}{c}{\\textbf{Offensive lang.}} & \\multicolumn{2}{c}{\\textbf{Hate speech}} \\\\",
  "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
  " & Human & LLM & Human & LLM \\\\"))

readr::write_csv(
  purrr::map_dfr(tabrows, ~ as.data.frame(.x)),
  file.path(OUT, "human_llm_components.csv"))
readr::write_csv(ver_prev, file.path(OUT, "kern_version_prevalence.csv"))
say("  wrote human_llm_components.csv, kern_version_prevalence.csv")
say("\nMajority label differs across versions for %.1f%% of tweets on OL, %.1f%% on HS.",
    flip_share[["OL"]] * 100, flip_share[["HS"]] * 100)

say("\nDone.")
