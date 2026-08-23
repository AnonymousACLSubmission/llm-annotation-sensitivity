# ---------------------------------------------------------------------------
# The pooled models used to be fit inside one big run_pooled_analysis() call.
# For the logit (binomial) family, the model x interaction glmer fits are heavy
# enough that they can crash the R session; splitting them into their own
# functions lets each be run (and saved) in its own Rmd chunk, so a crash in
# the interaction fits doesn't take the already-fit base/main models with it.
# fit_pooled_base() -> fit_pooled_interaction() / fit_pooled_design_interaction()
# -> combine_pooled_results() reproduce exactly what run_pooled_analysis() used
# to do in one shot; run_pooled_analysis() itself is now just a wrapper that
# chains them, kept for the gaussian/LPM path which isn't affected.
# ---------------------------------------------------------------------------

.pooled_glmer_ctrl <- function(family, n_cores, glmer_optCtrl) {
  if (family != "binomial") {
    return(list(ctrl = NULL, cleanup = function() {}))
  }

  library(optimParallel)
  cl <- parallel::makeCluster(n_cores)
  parallel::setDefaultCluster(cl)

  # lme4 requires the optimizer's return value to carry a name starting
  # with "fval"; optimParallel() just passes through stats::optim()'s
  # output ($value, $convergence), which fails that check. Wrap it so the
  # fields lme4 looks for exist.
  optimParallel_lme4 <- function(fn, par, lower, upper, control) {
    opt <- optimParallel::optimParallel(
      par = par, fn = fn, lower = lower, upper = upper, control = control
    )
    opt$fval <- opt$value
    opt$conv <- opt$convergence
    opt
  }

  ctrl <- glmerControl(
    optimizer   = optimParallel_lme4,
    calc.derivs = FALSE,
    optCtrl     = glmer_optCtrl
  )

  list(
    ctrl = ctrl,
    cleanup = function() {
      parallel::setDefaultCluster(NULL)
      parallel::stopCluster(cl)
    }
  )
}

# shared by NoModel/Main/Interaction: all three code batch/conf as factors and
# task as intercept + ol_first/hs_first; the ":model" case only fires for Interaction
.pooled_extract_fixed <- function(model, outcome) {
  broom.mixed::tidy(model, effects = "fixed") %>%
    mutate(
      outcome = outcome,
      stars = case_when(
        p.value < .001 ~ "***",
        p.value < .01  ~ "**",
        p.value < .05  ~ "*",
        TRUE           ~ ""
      ),
      term_clean = case_when(
        term == "ol_first"    ~ "Joint: OL first (vs. Separate Labeling)",
        term == "hs_first"    ~ "Joint: HS first (vs. Separate Labeling)",
        term == "batch1"      ~ "Batch Prompt (vs. Indiv. Prompts)",
        term == "conf1"       ~ "With Confidence (vs. Without)",
        str_detect(term, ":model") ~ paste0(
          str_remove(term, "^(ol_first|hs_first):model"), " x Asked First"
        ),
        str_starts(term, "model") ~ str_remove(term, "^model"),
        TRUE ~ term
      )
    ) %>%
    select(outcome, term_clean, estimate, std.error, statistic, p.value, stars)
}

.pooled_extract_model_stats <- function(model, outcome, family) {

  vc <- as.data.frame(VarCorr(model))
  sigma2 <- if (family == "binomial") (pi^2) / 3 else sigma(model)^2
  tau00 <- vc %>% filter(grp == "tweet_id") %>% pull(vcov)
  icc <- tau00 / (tau00 + sigma2)
  r2 <- performance::r2_nakagawa(model)

  tibble(
    outcome = outcome,
    sigma2 = sigma2,
    tau00_tweet_id = tau00,
    ICC = icc,
    N_tweet_id = n_distinct(model.frame(model)[["tweet_id"]]),
    Observations = nobs(model),
    Marginal_R2 = as.numeric(r2$R2_marginal),
    Conditional_R2 = as.numeric(r2$R2_conditional),
    logLik = as.numeric(logLik(model)),
    AIC = AIC(model),
    BIC = BIC(model)
  )
}

# ---- base/main models (NoModel + Main): cheap, always needed by the other fits ----

fit_pooled_base <- function(
    model_dfs,
    model_labels,
    reference_model = "gpt4o_mini",
    family = c("gaussian", "binomial"),
    n_cores = min(4, max(1, parallel::detectCores() - 1)),
    glmer_optCtrl = list(maxit = 200, factr = 1e8)
) {

  family <- match.arg(family)

  library(tidyverse)
  library(stringr)
  library(lme4)
  library(lmerTest)
  library(broom.mixed)
  library(performance)

  cluster_setup <- .pooled_glmer_ctrl(family, n_cores, glmer_optCtrl)
  glmer_ctrl <- cluster_setup$ctrl
  on.exit(cluster_setup$cleanup(), add = TRUE)

  model_keys <- c(reference_model, setdiff(names(model_dfs), reference_model))

  df_all <- imap_dfr(model_dfs, function(df, key) {
    df %>%
      select(batch_id, tweet_in_batch, tweet_id, tweet, condition, shot, starts_with("R")) %>%
      pivot_longer(
        cols      = starts_with("R"),
        names_to  = c("responder", ".value"),
        names_sep = "_"
      ) %>%
      mutate(
        OL    = as.integer(str_detect(label, "(?<!N)OL")),
        HS    = as.integer(str_detect(label, "(?<!N)HS")),
        batch = ifelse(grepl("batch", condition), 1, 0),
        conf  = ifelse(grepl("conf", condition), 1, 0),
        task  = case_when(
          grepl("C\\.OL", condition) ~ "C.OL",
          grepl("C\\.HS", condition) ~ "C.HS",
          grepl("^A", condition)     ~ "A",
          grepl("^B", condition)     ~ "B",
          TRUE                       ~ NA_character_
        ),
        model = key
      )
  })

  df_all <- df_all %>%
    mutate(model = factor(model, levels = model_keys, labels = model_labels[model_keys]))

  # task is coded as ol_first/hs_first (vs. Separate Labeling) throughout, so all specs
  # below share one representation: an intercept for Separate Labeling plus deviations.
  make_task_df <- function(outcome) {
    drop_task <- if (outcome == "OL") "C\\.HS" else "C\\.OL"

    df_all %>%
      filter(!grepl(drop_task, condition)) %>%
      mutate(
        task = factor(
          task,
          levels = if (outcome == "OL") c("A", "B", "C.OL") else c("A", "B", "C.HS"),
          labels = c("Joint: OL first", "Joint: HS first", "Separate Labeling")
        ),
        ol_first = as.integer(task == "Joint: OL first"),
        hs_first = as.integer(task == "Joint: HS first")
      )
  }

  fit_no_model <- function(outcome) {

    df_out <- make_task_df(outcome) %>%
      mutate(batch = factor(batch), conf = factor(conf))

    fml <- as.formula(paste0(outcome, " ~ ol_first + hs_first + batch + conf + (1 | tweet_id)"))
    if (family == "binomial") {
      glmer(fml, data = df_out, family = binomial(link = "logit"), verbose = 1, control = glmer_ctrl)
    } else {
      lmer(fml, data = df_out, verbose = 1)
    }
  }

  fit_one <- function(outcome) {

    df_out <- make_task_df(outcome) %>%
      mutate(batch = factor(batch), conf = factor(conf))

    fml <- as.formula(paste0(outcome, " ~ ol_first + hs_first + batch + conf + model + (1 | tweet_id)"))
    if (family == "binomial") {
      glmer(fml, data = df_out, family = binomial(link = "logit"), verbose = 1, control = glmer_ctrl)
    } else {
      lmer(fml, data = df_out, verbose = 1)
    }
  }

  model_OL_nomodel <- fit_no_model("OL")
  model_HS_nomodel <- fit_no_model("HS")

  model_OL <- fit_one("OL")
  model_HS <- fit_one("HS")

  reference_label <- model_labels[[reference_model]]

  nomodel_fixed_table <- bind_rows(
    .pooled_extract_fixed(model_OL_nomodel, "OL"),
    .pooled_extract_fixed(model_HS_nomodel, "HS")
  )

  fixed_table <- bind_rows(
    .pooled_extract_fixed(model_OL, "OL"),
    .pooled_extract_fixed(model_HS, "HS")
  )

  model_level_labels <- unname(model_labels[setdiff(model_keys, reference_model)])

  row_order <- c(
    "(Intercept)",
    "Joint: OL first (vs. Separate Labeling)",
    "Joint: HS first (vs. Separate Labeling)",
    "With Confidence (vs. Without)",
    "Batch Prompt (vs. Indiv. Prompts)",
    model_level_labels
  )

  paper_table <- fixed_table %>%
    mutate(
      estimate_fmt = ifelse(
        is.na(std.error),
        sprintf("%.3f", estimate),
        sprintf("%.3f%s", estimate, stars)
      ),
      se_fmt = ifelse(is.na(std.error), "--", sprintf("%.3f", std.error))
    ) %>%
    select(term_clean, outcome, estimate_fmt, se_fmt) %>%
    pivot_wider(
      names_from = outcome,
      values_from = c(estimate_fmt, se_fmt),
      names_glue = "{outcome}_{.value}"
    ) %>%
    select(term_clean, OL_estimate_fmt, OL_se_fmt, HS_estimate_fmt, HS_se_fmt) %>%
    mutate(term_clean = factor(term_clean, levels = row_order)) %>%
    arrange(term_clean) %>%
    mutate(term_clean = as.character(term_clean))

  list(
    model_dfs = model_dfs,
    model_labels = model_labels,
    reference_model = reference_model,
    family = family,
    n_cores = n_cores,
    glmer_optCtrl = glmer_optCtrl,
    model_keys = model_keys,
    model_level_labels = model_level_labels,
    reference_label = reference_label,
    df_all = df_all,
    make_task_df = make_task_df,
    model_OL_nomodel = model_OL_nomodel,
    model_HS_nomodel = model_HS_nomodel,
    model_OL = model_OL,
    model_HS = model_HS,
    nomodel_fixed_table = nomodel_fixed_table,
    fixed_table = fixed_table,
    paper_table = paper_table
  )
}

# ---- task-structure interaction model: model x Asked First, fit for both families ----
# (heterogeneity test for the task-order effect; superseded by random slopes for
# the LPM, but reported here too since logit gets the same spec now). This is one
# of the two heavy "with interactions" fits - kept in its own function/chunk since
# it's a plausible crash point for the binomial family.

fit_pooled_interaction <- function(base, n_cores = base$n_cores) {

  family <- base$family
  make_task_df <- base$make_task_df
  model_level_labels <- base$model_level_labels

  library(tidyverse)
  library(stringr)
  library(lme4)
  library(lmerTest)
  library(broom.mixed)

  cluster_setup <- .pooled_glmer_ctrl(family, n_cores, base$glmer_optCtrl)
  glmer_ctrl <- cluster_setup$ctrl
  on.exit(cluster_setup$cleanup(), add = TRUE)

  fit_interaction <- function(outcome) {
    interaction_var <- if (outcome == "OL") "ol_first" else "hs_first"

    df_out <- make_task_df(outcome) %>%
      mutate(batch = factor(batch), conf = factor(conf))

    fml <- as.formula(paste0(
      outcome, " ~ ol_first + hs_first + batch + conf + model + ",
      interaction_var, ":model + (1 | tweet_id)"
    ))
    if (family == "binomial") {
      glmer(fml, data = df_out, family = binomial(link = "logit"), verbose = 1, control = glmer_ctrl)
    } else {
      lmer(fml, data = df_out, verbose = 1)
    }
  }

  model_OL_interaction <- fit_interaction("OL")
  model_HS_interaction <- fit_interaction("HS")

  interaction_fixed_table <- bind_rows(
    .pooled_extract_fixed(model_OL_interaction, "OL"),
    .pooled_extract_fixed(model_HS_interaction, "HS")
  )

  interaction_row_order <- c(
    "(Intercept)",
    "Joint: OL first (vs. Separate Labeling)",
    "Joint: HS first (vs. Separate Labeling)",
    "With Confidence (vs. Without)",
    "Batch Prompt (vs. Indiv. Prompts)",
    model_level_labels,
    paste0(model_level_labels, " x Asked First")
  )

  interaction_paper_table <- interaction_fixed_table %>%
    mutate(
      estimate_fmt = sprintf("%.3f%s", estimate, stars),
      se_fmt = sprintf("%.3f", std.error)
    ) %>%
    select(term_clean, outcome, estimate_fmt, se_fmt) %>%
    pivot_wider(
      names_from = outcome,
      values_from = c(estimate_fmt, se_fmt),
      names_glue = "{outcome}_{.value}"
    ) %>%
    select(term_clean, OL_estimate_fmt, OL_se_fmt, HS_estimate_fmt, HS_se_fmt) %>%
    mutate(term_clean = factor(term_clean, levels = interaction_row_order)) %>%
    arrange(term_clean) %>%
    mutate(term_clean = as.character(term_clean))

  list(
    model_OL_interaction = model_OL_interaction,
    model_HS_interaction = model_HS_interaction,
    interaction_fixed_table = interaction_fixed_table,
    interaction_paper_table = interaction_paper_table
  )
}

# ---- design-interaction model: model x Batch and model x Confidence (no task-structure
# interaction), fit for both families - this is the reviewer-flagged heterogeneity
# check (batch/confidence direction flips across models), kept separate from the
# task-structure interaction above since including all 4 vars x model at once would
# roughly double the added parameters for no added test of the flagged concern. This is
# the other heavy "with interactions" fit, also kept in its own function/chunk. ----

fit_pooled_design_interaction <- function(base, n_cores = base$n_cores) {

  family <- base$family
  make_task_df <- base$make_task_df
  model_level_labels <- base$model_level_labels

  library(tidyverse)
  library(stringr)
  library(lme4)
  library(lmerTest)
  library(broom.mixed)

  cluster_setup <- .pooled_glmer_ctrl(family, n_cores, base$glmer_optCtrl)
  glmer_ctrl <- cluster_setup$ctrl
  on.exit(cluster_setup$cleanup(), add = TRUE)

  fit_design_interaction <- function(outcome) {

    df_out <- make_task_df(outcome) %>%
      mutate(batch = factor(batch), conf = factor(conf))

    fml <- as.formula(paste0(
      outcome, " ~ ol_first + hs_first + batch + conf + model + batch:model + conf:model + (1 | tweet_id)"
    ))
    if (family == "binomial") {
      glmer(fml, data = df_out, family = binomial(link = "logit"), verbose = 1, control = glmer_ctrl)
    } else {
      lmer(fml, data = df_out, verbose = 1)
    }
  }

  model_OL_designinteraction <- fit_design_interaction("OL")
  model_HS_designinteraction <- fit_design_interaction("HS")

  # batch/conf enter as factor(batch)/factor(conf) here (like NoModel/Main/Interaction),
  # so terms are "batch1:model..."/"conf1:model...", not "batch:model.../conf:model..."
  extract_fixed_designinteraction <- function(model, outcome) {
    broom.mixed::tidy(model, effects = "fixed") %>%
      mutate(
        outcome = outcome,
        stars = case_when(
          p.value < .001 ~ "***",
          p.value < .01  ~ "**",
          p.value < .05  ~ "*",
          TRUE           ~ ""
        ),
        term_clean = case_when(
          term == "ol_first"    ~ "Joint: OL first (vs. Separate Labeling)",
          term == "hs_first"    ~ "Joint: HS first (vs. Separate Labeling)",
          term == "batch1"      ~ "Batch Prompt (vs. Indiv. Prompts)",
          term == "conf1"       ~ "With Confidence (vs. Without)",
          str_detect(term, "^batch1:model") ~ paste0(
            str_remove(term, "^batch1:model"), " x Batch Prompt"
          ),
          str_detect(term, "^conf1:model") ~ paste0(
            str_remove(term, "^conf1:model"), " x With Confidence"
          ),
          str_starts(term, "model") ~ str_remove(term, "^model"),
          TRUE ~ term
        )
      ) %>%
      select(outcome, term_clean, estimate, std.error, statistic, p.value, stars)
  }

  designinteraction_fixed_table <- bind_rows(
    extract_fixed_designinteraction(model_OL_designinteraction, "OL"),
    extract_fixed_designinteraction(model_HS_designinteraction, "HS")
  )

  designinteraction_row_order <- c(
    "(Intercept)",
    "Joint: OL first (vs. Separate Labeling)",
    "Joint: HS first (vs. Separate Labeling)",
    "With Confidence (vs. Without)",
    "Batch Prompt (vs. Indiv. Prompts)",
    model_level_labels,
    paste0(model_level_labels, " x Batch Prompt"),
    paste0(model_level_labels, " x With Confidence")
  )

  designinteraction_paper_table <- designinteraction_fixed_table %>%
    mutate(
      estimate_fmt = sprintf("%.3f%s", estimate, stars),
      se_fmt = sprintf("%.3f", std.error)
    ) %>%
    select(term_clean, outcome, estimate_fmt, se_fmt) %>%
    pivot_wider(
      names_from = outcome,
      values_from = c(estimate_fmt, se_fmt),
      names_glue = "{outcome}_{.value}"
    ) %>%
    select(term_clean, OL_estimate_fmt, OL_se_fmt, HS_estimate_fmt, HS_se_fmt) %>%
    mutate(term_clean = factor(term_clean, levels = designinteraction_row_order)) %>%
    arrange(term_clean) %>%
    mutate(term_clean = as.character(term_clean))

  list(
    model_OL_designinteraction = model_OL_designinteraction,
    model_HS_designinteraction = model_HS_designinteraction,
    designinteraction_fixed_table = designinteraction_fixed_table,
    designinteraction_paper_table = designinteraction_paper_table
  )
}

# ---- random-slopes model (gaussian/LPM only) ----

fit_pooled_random_slopes <- function(base) {

  stopifnot(base$family == "gaussian")

  make_task_df <- base$make_task_df
  model_level_labels <- base$model_level_labels
  nomodel_fixed_table <- base$nomodel_fixed_table
  fixed_table <- base$fixed_table
  model_OL_nomodel <- base$model_OL_nomodel
  model_HS_nomodel <- base$model_HS_nomodel
  model_OL <- base$model_OL
  model_HS <- base$model_HS

  library(tidyverse)
  library(stringr)
  library(lme4)
  library(lmerTest)
  library(broom.mixed)
  library(performance)

  fit_random_slopes <- function(outcome) {

    df_out <- make_task_df(outcome)

    # `model` is both a fixed effect (its own average level, like Main) and the random-slopes
    # grouping factor; the random term is slopes-only (0 +, no intercept) so it isn't redundant
    # with the fixed model effect - it captures how much each model's own slope deviates,
    # net of that model's average level.
    fml <- as.formula(paste0(
      outcome, " ~ ol_first + hs_first + batch + conf + model + ",
      "(0 + ol_first + hs_first + batch + conf || model) + (1 | tweet_id)"
    ))

    lmer(fml, data = df_out, verbose = 1)
  }

  model_OL_randomslopes <- fit_random_slopes("OL")
  model_HS_randomslopes <- fit_random_slopes("HS")

  # random-slopes keeps batch/conf numeric (0/1), so its terms are "batch"/"conf", not "batch1"/"conf1"
  extract_fixed_randomslopes <- function(model, outcome) {
    broom.mixed::tidy(model, effects = "fixed") %>%
      mutate(
        outcome = outcome,
        stars = case_when(
          p.value < .001 ~ "***",
          p.value < .01  ~ "**",
          p.value < .05  ~ "*",
          TRUE           ~ ""
        ),
        term_clean = case_when(
          term == "ol_first"    ~ "Joint: OL first (vs. Separate Labeling)",
          term == "hs_first"    ~ "Joint: HS first (vs. Separate Labeling)",
          term == "batch"       ~ "Batch Prompt (vs. Indiv. Prompts)",
          term == "conf"        ~ "With Confidence (vs. Without)",
          str_starts(term, "model") ~ str_remove(term, "^model"),
          TRUE ~ term
        )
      ) %>%
      select(outcome, term_clean, estimate, std.error, statistic, p.value, stars)
  }

  randomslopes_fixed_table <- bind_rows(
    extract_fixed_randomslopes(model_OL_randomslopes, "OL"),
    extract_fixed_randomslopes(model_HS_randomslopes, "HS")
  )

  combined_row_order_rs <- c(
    "(Intercept)",
    "Joint: OL first (vs. Separate Labeling)",
    "Joint: HS first (vs. Separate Labeling)",
    "With Confidence (vs. Without)",
    "Batch Prompt (vs. Indiv. Prompts)",
    model_level_labels
  )

  spec_levels_rs <- c("OL_NoModel", "OL_Main", "OL_RandomSlopes", "HS_NoModel", "HS_Main", "HS_RandomSlopes")
  spec_cols_rs <- c(
    "OL_NoModel_estimate_fmt", "OL_NoModel_se_fmt",
    "OL_Main_estimate_fmt", "OL_Main_se_fmt",
    "OL_RandomSlopes_estimate_fmt", "OL_RandomSlopes_se_fmt",
    "HS_NoModel_estimate_fmt", "HS_NoModel_se_fmt",
    "HS_Main_estimate_fmt", "HS_Main_se_fmt",
    "HS_RandomSlopes_estimate_fmt", "HS_RandomSlopes_se_fmt"
  )

  combined_long_rs <- bind_rows(
    nomodel_fixed_table %>% filter(outcome == "OL") %>% mutate(spec = "OL_NoModel"),
    fixed_table %>% filter(outcome == "OL") %>% mutate(spec = "OL_Main"),
    randomslopes_fixed_table %>% filter(outcome == "OL") %>% mutate(spec = "OL_RandomSlopes"),
    nomodel_fixed_table %>% filter(outcome == "HS") %>% mutate(spec = "HS_NoModel"),
    fixed_table %>% filter(outcome == "HS") %>% mutate(spec = "HS_Main"),
    randomslopes_fixed_table %>% filter(outcome == "HS") %>% mutate(spec = "HS_RandomSlopes")
  ) %>%
    mutate(
      spec = factor(spec, levels = spec_levels_rs),
      estimate_fmt = ifelse(
        is.na(std.error),
        sprintf("%.3f", estimate),
        sprintf("%.3f%s", estimate, stars)
      ),
      se_fmt = ifelse(is.na(std.error), "--", sprintf("%.3f", std.error))
    )

  combined_coef_table_rs <- combined_long_rs %>%
    select(term_clean, spec, estimate_fmt, se_fmt) %>%
    pivot_wider(
      names_from = spec,
      values_from = c(estimate_fmt, se_fmt),
      names_glue = "{spec}_{.value}",
      values_fill = "--"
    ) %>%
    select(term_clean, all_of(spec_cols_rs))

  combined_stats_table_rs <- bind_rows(
    .pooled_extract_model_stats(model_OL_nomodel, "OL", "gaussian") %>% mutate(spec = "OL_NoModel"),
    .pooled_extract_model_stats(model_OL, "OL", "gaussian") %>% mutate(spec = "OL_Main"),
    .pooled_extract_model_stats(model_OL_randomslopes, "OL", "gaussian") %>% mutate(spec = "OL_RandomSlopes"),
    .pooled_extract_model_stats(model_HS_nomodel, "HS", "gaussian") %>% mutate(spec = "HS_NoModel"),
    .pooled_extract_model_stats(model_HS, "HS", "gaussian") %>% mutate(spec = "HS_Main"),
    .pooled_extract_model_stats(model_HS_randomslopes, "HS", "gaussian") %>% mutate(spec = "HS_RandomSlopes")
  ) %>%
    pivot_longer(
      cols = c(sigma2, tau00_tweet_id, ICC, N_tweet_id, Observations, Marginal_R2, Conditional_R2, logLik, AIC, BIC),
      names_to = "term_clean",
      values_to = "value"
    ) %>%
    mutate(
      spec = factor(spec, levels = spec_levels_rs),
      estimate_fmt = case_when(
        term_clean %in% c("N_tweet_id", "Observations") ~ format(round(value), big.mark = ",", scientific = FALSE),
        term_clean %in% c("logLik", "AIC", "BIC") ~ sprintf("%.1f", value),
        TRUE ~ sprintf("%.3f", value)
      ),
      se_fmt = "--",
      term_clean = recode(
        term_clean,
        "sigma2" = "$\\sigma^2$",
        "tau00_tweet_id" = "$\\tau_{00}$\\textsubscript{tweet\\_id}",
        "N_tweet_id" = "$N$\\textsubscript{tweet\\_id}",
        "Marginal_R2" = "Marginal $R^2$",
        "Conditional_R2" = "Conditional $R^2$"
      )
    ) %>%
    select(term_clean, spec, estimate_fmt, se_fmt) %>%
    pivot_wider(
      names_from = spec,
      values_from = c(estimate_fmt, se_fmt),
      names_glue = "{spec}_{.value}",
      values_fill = "--"
    ) %>%
    select(term_clean, all_of(spec_cols_rs))

  # 4 uncorrelated random-slope variance components (across `model`); no random intercept
  # since the model-level average is already captured by the fixed `model` effect above
  variance_row_order <- c(
    "Var: Joint OL first",
    "Var: Joint HS first",
    "Var: Batch Prompt",
    "Var: With Confidence"
  )

  extract_variance_components <- function(model, outcome) {
    as.data.frame(VarCorr(model)) %>%
      filter(str_starts(grp, "model"), is.na(var2)) %>%
      transmute(
        outcome = outcome,
        term_clean = recode(
          var1,
          "ol_first"    = "Var: Joint OL first",
          "hs_first"    = "Var: Joint HS first",
          "batch"       = "Var: Batch Prompt",
          "conf"        = "Var: With Confidence"
        ),
        estimate_fmt = sprintf("%.4f", vcov),
        se_fmt       = sprintf("%.4f", sqrt(vcov))
      )
  }

  variance_table_rs <- bind_rows(
    extract_variance_components(model_OL_randomslopes, "OL") %>% mutate(spec = "OL_RandomSlopes"),
    extract_variance_components(model_HS_randomslopes, "HS") %>% mutate(spec = "HS_RandomSlopes")
  ) %>%
    select(term_clean, spec, estimate_fmt, se_fmt) %>%
    pivot_wider(
      names_from = spec,
      values_from = c(estimate_fmt, se_fmt),
      names_glue = "{spec}_{.value}"
    ) %>%
    mutate(
      OL_NoModel_estimate_fmt = "--", OL_NoModel_se_fmt = "--",
      OL_Main_estimate_fmt = "--", OL_Main_se_fmt = "--",
      HS_NoModel_estimate_fmt = "--", HS_NoModel_se_fmt = "--",
      HS_Main_estimate_fmt = "--", HS_Main_se_fmt = "--"
    ) %>%
    select(term_clean, all_of(spec_cols_rs))

  # LRT of RandomSlopes vs. Main: since the two share the exact same fixed-effects
  # formula and differ only in the random-effects structure (the 4 uncorrelated
  # model-level slopes), the REML fits are directly comparable and refit=FALSE
  # avoids the (here unnecessary) cost of refitting both models under ML.
  extract_lrt_vs_main <- function(model_main, model_randomslopes, outcome) {
    lrt <- suppressMessages(anova(model_main, model_randomslopes, refit = FALSE))
    tibble(
      outcome = outcome,
      chisq = lrt$Chisq[2],
      df    = lrt$Df[2],
      p     = lrt[["Pr(>Chisq)"]][2]
    )
  }

  lrt_table_rs <- bind_rows(
    extract_lrt_vs_main(model_OL, model_OL_randomslopes, "OL"),
    extract_lrt_vs_main(model_HS, model_HS_randomslopes, "HS")
  ) %>%
    mutate(
      chisq_fmt = sprintf("%.2f", chisq),
      p_fmt     = ifelse(p < .001, "<.001", sprintf("%.3f", p))
    )

  # df is a structural property of the two nested formulas (4 uncorrelated random-slope
  # variances added), so it's identical for OL/HS and gets folded into the chi-sq row label
  # rather than reported as its own line.
  stopifnot(length(unique(lrt_table_rs$df)) == 1)
  lrt_chisq_label <- paste0("LRT $\\chi^2_{", unique(lrt_table_rs$df), "}$ (vs. Models)")

  lrt_row_order <- c(
    lrt_chisq_label,
    "LRT $p$"
  )

  lrt_long_rs <- bind_rows(
    lrt_table_rs %>% transmute(outcome, term_clean = lrt_chisq_label, estimate_fmt = chisq_fmt),
    lrt_table_rs %>% transmute(outcome, term_clean = "LRT $p$",       estimate_fmt = p_fmt)
  ) %>%
    mutate(spec = paste0(outcome, "_RandomSlopes"), se_fmt = "--")

  lrt_table_rs_wide <- lrt_long_rs %>%
    select(term_clean, spec, estimate_fmt, se_fmt) %>%
    pivot_wider(
      names_from = spec,
      values_from = c(estimate_fmt, se_fmt),
      names_glue = "{spec}_{.value}"
    ) %>%
    mutate(
      OL_NoModel_estimate_fmt = "--", OL_NoModel_se_fmt = "--",
      OL_Main_estimate_fmt = "--", OL_Main_se_fmt = "--",
      HS_NoModel_estimate_fmt = "--", HS_NoModel_se_fmt = "--",
      HS_Main_estimate_fmt = "--", HS_Main_se_fmt = "--"
    ) %>%
    select(term_clean, all_of(spec_cols_rs))

  stats_row_order <- c(
    "$\\sigma^2$",
    "$\\tau_{00}$\\textsubscript{tweet\\_id}",
    "ICC",
    "$N$\\textsubscript{tweet\\_id}",
    "Observations",
    "Marginal $R^2$",
    "Conditional $R^2$",
    "logLik",
    "AIC",
    "BIC"
  )

  combined_paper_table_randomslopes <- bind_rows(
    combined_coef_table_rs, combined_stats_table_rs, variance_table_rs, lrt_table_rs_wide
  ) %>%
    mutate(term_clean = factor(term_clean, levels = c(combined_row_order_rs, stats_row_order, variance_row_order, lrt_row_order))) %>%
    arrange(term_clean) %>%
    mutate(term_clean = as.character(term_clean))

  list(
    model_OL_randomslopes = model_OL_randomslopes,
    model_HS_randomslopes = model_HS_randomslopes,
    combined_paper_table_randomslopes = combined_paper_table_randomslopes
  )
}

# ---- combine base + interaction + design-interaction (+ optional random-slopes)
# into the same result shape run_pooled_analysis() used to return in one call ----

combine_pooled_results <- function(base, interaction, design_interaction, random_slopes = NULL) {

  library(tidyverse)
  library(stringr)

  family <- base$family
  model_level_labels <- base$model_level_labels
  nomodel_fixed_table <- base$nomodel_fixed_table
  fixed_table <- base$fixed_table

  model_OL_interaction <- interaction$model_OL_interaction
  model_HS_interaction <- interaction$model_HS_interaction
  interaction_fixed_table <- interaction$interaction_fixed_table

  model_OL_designinteraction <- design_interaction$model_OL_designinteraction
  model_HS_designinteraction <- design_interaction$model_HS_designinteraction
  designinteraction_fixed_table <- design_interaction$designinteraction_fixed_table

  # combined_paper_table has 4 spec columns per outcome (NoModel/Main/Interaction/
  # DesignInteraction) - same shape for both families now that both interaction
  # models run regardless of family. Interaction = model x Asked First (task
  # structure); DesignInteraction = model x Batch + model x Confidence.
  combined_row_order <- c(
    "(Intercept)",
    "Joint: OL first (vs. Separate Labeling)",
    "Joint: HS first (vs. Separate Labeling)",
    "With Confidence (vs. Without)",
    "Batch Prompt (vs. Indiv. Prompts)",
    model_level_labels,
    paste0(model_level_labels, " x Asked First"),
    paste0(model_level_labels, " x Batch Prompt"),
    paste0(model_level_labels, " x With Confidence")
  )

  spec_levels <- c(
    "OL_NoModel", "OL_Main", "OL_Interaction", "OL_DesignInteraction",
    "HS_NoModel", "HS_Main", "HS_Interaction", "HS_DesignInteraction"
  )
  spec_cols <- c(
    "OL_NoModel_estimate_fmt", "OL_NoModel_se_fmt",
    "OL_Main_estimate_fmt", "OL_Main_se_fmt",
    "OL_Interaction_estimate_fmt", "OL_Interaction_se_fmt",
    "OL_DesignInteraction_estimate_fmt", "OL_DesignInteraction_se_fmt",
    "HS_NoModel_estimate_fmt", "HS_NoModel_se_fmt",
    "HS_Main_estimate_fmt", "HS_Main_se_fmt",
    "HS_Interaction_estimate_fmt", "HS_Interaction_se_fmt",
    "HS_DesignInteraction_estimate_fmt", "HS_DesignInteraction_se_fmt"
  )

  combined_long <- bind_rows(
    nomodel_fixed_table %>% filter(outcome == "OL") %>% mutate(spec = "OL_NoModel"),
    fixed_table %>% filter(outcome == "OL") %>% mutate(spec = "OL_Main"),
    interaction_fixed_table %>% filter(outcome == "OL") %>% mutate(spec = "OL_Interaction"),
    designinteraction_fixed_table %>% filter(outcome == "OL") %>% mutate(spec = "OL_DesignInteraction"),
    nomodel_fixed_table %>% filter(outcome == "HS") %>% mutate(spec = "HS_NoModel"),
    fixed_table %>% filter(outcome == "HS") %>% mutate(spec = "HS_Main"),
    interaction_fixed_table %>% filter(outcome == "HS") %>% mutate(spec = "HS_Interaction"),
    designinteraction_fixed_table %>% filter(outcome == "HS") %>% mutate(spec = "HS_DesignInteraction")
  )

  stats_models <- bind_rows(
    .pooled_extract_model_stats(base$model_OL_nomodel, "OL", family) %>% mutate(spec = "OL_NoModel"),
    .pooled_extract_model_stats(base$model_OL, "OL", family) %>% mutate(spec = "OL_Main"),
    .pooled_extract_model_stats(model_OL_interaction, "OL", family) %>% mutate(spec = "OL_Interaction"),
    .pooled_extract_model_stats(model_OL_designinteraction, "OL", family) %>% mutate(spec = "OL_DesignInteraction"),
    .pooled_extract_model_stats(base$model_HS_nomodel, "HS", family) %>% mutate(spec = "HS_NoModel"),
    .pooled_extract_model_stats(base$model_HS, "HS", family) %>% mutate(spec = "HS_Main"),
    .pooled_extract_model_stats(model_HS_interaction, "HS", family) %>% mutate(spec = "HS_Interaction"),
    .pooled_extract_model_stats(model_HS_designinteraction, "HS", family) %>% mutate(spec = "HS_DesignInteraction")
  )

  combined_long <- combined_long %>%
    mutate(
      spec = factor(spec, levels = spec_levels),
      estimate_fmt = ifelse(
        is.na(std.error),
        sprintf("%.3f", estimate),
        sprintf("%.3f%s", estimate, stars)
      ),
      se_fmt = ifelse(is.na(std.error), "--", sprintf("%.3f", std.error))
    )

  combined_coef_table <- combined_long %>%
    select(term_clean, spec, estimate_fmt, se_fmt) %>%
    pivot_wider(
      names_from = spec,
      values_from = c(estimate_fmt, se_fmt),
      names_glue = "{spec}_{.value}",
      values_fill = "--"
    ) %>%
    select(term_clean, all_of(spec_cols))

  stats_row_order <- c(
    "$\\sigma^2$",
    "$\\tau_{00}$\\textsubscript{tweet\\_id}",
    "ICC",
    "$N$\\textsubscript{tweet\\_id}",
    "Observations",
    "Marginal $R^2$",
    "Conditional $R^2$",
    "logLik",
    "AIC",
    "BIC"
  )

  combined_stats_table <- stats_models %>%
    pivot_longer(
      cols = c(sigma2, tau00_tweet_id, ICC, N_tweet_id, Observations, Marginal_R2, Conditional_R2, logLik, AIC, BIC),
      names_to = "term_clean",
      values_to = "value"
    ) %>%
    mutate(
      spec = factor(spec, levels = spec_levels),
      estimate_fmt = case_when(
        term_clean %in% c("N_tweet_id", "Observations") ~ format(round(value), big.mark = ",", scientific = FALSE),
        term_clean %in% c("logLik", "AIC", "BIC") ~ sprintf("%.1f", value),
        TRUE ~ sprintf("%.3f", value)
      ),
      se_fmt = "--",
      term_clean = recode(
        term_clean,
        "sigma2" = "$\\sigma^2$",
        "tau00_tweet_id" = "$\\tau_{00}$\\textsubscript{tweet\\_id}",
        "N_tweet_id" = "$N$\\textsubscript{tweet\\_id}",
        "Marginal_R2" = "Marginal $R^2$",
        "Conditional_R2" = "Conditional $R^2$"
      )
    ) %>%
    select(term_clean, spec, estimate_fmt, se_fmt) %>%
    pivot_wider(
      names_from = spec,
      values_from = c(estimate_fmt, se_fmt),
      names_glue = "{spec}_{.value}",
      values_fill = "--"
    ) %>%
    select(term_clean, all_of(spec_cols))

  combined_paper_table <- bind_rows(combined_coef_table, combined_stats_table) %>%
    mutate(term_clean = factor(term_clean, levels = c(combined_row_order, stats_row_order))) %>%
    arrange(term_clean) %>%
    mutate(term_clean = as.character(term_clean))

  model_OL_randomslopes <- NULL
  model_HS_randomslopes <- NULL
  combined_paper_table_randomslopes <- NULL
  if (!is.null(random_slopes)) {
    model_OL_randomslopes <- random_slopes$model_OL_randomslopes
    model_HS_randomslopes <- random_slopes$model_HS_randomslopes
    combined_paper_table_randomslopes <- random_slopes$combined_paper_table_randomslopes
  }

  list(
    model_OL_nomodel = base$model_OL_nomodel,
    model_HS_nomodel = base$model_HS_nomodel,
    nomodel_fixed_table = nomodel_fixed_table,
    model_OL = base$model_OL,
    model_HS = base$model_HS,
    fixed_table = fixed_table,
    paper_table = base$paper_table,
    model_OL_interaction = model_OL_interaction,
    model_HS_interaction = model_HS_interaction,
    interaction_fixed_table = interaction_fixed_table,
    interaction_paper_table = interaction$interaction_paper_table,
    model_OL_designinteraction = model_OL_designinteraction,
    model_HS_designinteraction = model_HS_designinteraction,
    designinteraction_fixed_table = designinteraction_fixed_table,
    designinteraction_paper_table = design_interaction$designinteraction_paper_table,
    combined_paper_table = combined_paper_table,
    model_OL_randomslopes = model_OL_randomslopes,
    model_HS_randomslopes = model_HS_randomslopes,
    combined_paper_table_randomslopes = combined_paper_table_randomslopes,
    model_level_labels = model_level_labels,
    reference_model = base$reference_model,
    reference_label = base$reference_label
  )
}

# base-only variant: NoModel + Main specs, no model x interaction terms at all, so
# it only needs fit_pooled_base() and never touches the crash-prone interaction fits.
combine_pooled_results_base_only <- function(base) {

  library(tidyverse)
  library(stringr)
  library(lme4)
  library(broom.mixed)
  library(performance)

  family <- base$family
  model_level_labels <- base$model_level_labels
  nomodel_fixed_table <- base$nomodel_fixed_table
  fixed_table <- base$fixed_table

  combined_row_order <- c(
    "(Intercept)",
    "Joint: OL first (vs. Separate Labeling)",
    "Joint: HS first (vs. Separate Labeling)",
    "With Confidence (vs. Without)",
    "Batch Prompt (vs. Indiv. Prompts)",
    model_level_labels
  )

  spec_levels <- c("OL_NoModel", "OL_Main", "HS_NoModel", "HS_Main")
  spec_cols <- c(
    "OL_NoModel_estimate_fmt", "OL_NoModel_se_fmt",
    "OL_Main_estimate_fmt", "OL_Main_se_fmt",
    "HS_NoModel_estimate_fmt", "HS_NoModel_se_fmt",
    "HS_Main_estimate_fmt", "HS_Main_se_fmt"
  )

  combined_long <- bind_rows(
    nomodel_fixed_table %>% filter(outcome == "OL") %>% mutate(spec = "OL_NoModel"),
    fixed_table %>% filter(outcome == "OL") %>% mutate(spec = "OL_Main"),
    nomodel_fixed_table %>% filter(outcome == "HS") %>% mutate(spec = "HS_NoModel"),
    fixed_table %>% filter(outcome == "HS") %>% mutate(spec = "HS_Main")
  )

  stats_models <- bind_rows(
    .pooled_extract_model_stats(base$model_OL_nomodel, "OL", family) %>% mutate(spec = "OL_NoModel"),
    .pooled_extract_model_stats(base$model_OL, "OL", family) %>% mutate(spec = "OL_Main"),
    .pooled_extract_model_stats(base$model_HS_nomodel, "HS", family) %>% mutate(spec = "HS_NoModel"),
    .pooled_extract_model_stats(base$model_HS, "HS", family) %>% mutate(spec = "HS_Main")
  )

  combined_long <- combined_long %>%
    mutate(
      spec = factor(spec, levels = spec_levels),
      estimate_fmt = ifelse(
        is.na(std.error),
        sprintf("%.3f", estimate),
        sprintf("%.3f%s", estimate, stars)
      ),
      se_fmt = ifelse(is.na(std.error), "--", sprintf("%.3f", std.error))
    )

  combined_coef_table <- combined_long %>%
    select(term_clean, spec, estimate_fmt, se_fmt) %>%
    pivot_wider(
      names_from = spec,
      values_from = c(estimate_fmt, se_fmt),
      names_glue = "{spec}_{.value}",
      values_fill = "--"
    ) %>%
    select(term_clean, all_of(spec_cols))

  stats_row_order <- c(
    "$\\sigma^2$",
    "$\\tau_{00}$\\textsubscript{tweet\\_id}",
    "ICC",
    "$N$\\textsubscript{tweet\\_id}",
    "Observations",
    "Marginal $R^2$",
    "Conditional $R^2$",
    "logLik",
    "AIC",
    "BIC"
  )

  combined_stats_table <- stats_models %>%
    pivot_longer(
      cols = c(sigma2, tau00_tweet_id, ICC, N_tweet_id, Observations, Marginal_R2, Conditional_R2, logLik, AIC, BIC),
      names_to = "term_clean",
      values_to = "value"
    ) %>%
    mutate(
      spec = factor(spec, levels = spec_levels),
      estimate_fmt = case_when(
        term_clean %in% c("N_tweet_id", "Observations") ~ format(round(value), big.mark = ",", scientific = FALSE),
        term_clean %in% c("logLik", "AIC", "BIC") ~ sprintf("%.1f", value),
        TRUE ~ sprintf("%.3f", value)
      ),
      se_fmt = "--",
      term_clean = recode(
        term_clean,
        "sigma2" = "$\\sigma^2$",
        "tau00_tweet_id" = "$\\tau_{00}$\\textsubscript{tweet\\_id}",
        "N_tweet_id" = "$N$\\textsubscript{tweet\\_id}",
        "Marginal_R2" = "Marginal $R^2$",
        "Conditional_R2" = "Conditional $R^2$"
      )
    ) %>%
    select(term_clean, spec, estimate_fmt, se_fmt) %>%
    pivot_wider(
      names_from = spec,
      values_from = c(estimate_fmt, se_fmt),
      names_glue = "{spec}_{.value}",
      values_fill = "--"
    ) %>%
    select(term_clean, all_of(spec_cols))

  combined_paper_table <- bind_rows(combined_coef_table, combined_stats_table) %>%
    mutate(term_clean = factor(term_clean, levels = c(combined_row_order, stats_row_order))) %>%
    arrange(term_clean) %>%
    mutate(term_clean = as.character(term_clean))

  list(
    model_OL_nomodel = base$model_OL_nomodel,
    model_HS_nomodel = base$model_HS_nomodel,
    nomodel_fixed_table = nomodel_fixed_table,
    model_OL = base$model_OL,
    model_HS = base$model_HS,
    fixed_table = fixed_table,
    paper_table = base$paper_table,
    combined_paper_table = combined_paper_table,
    model_level_labels = model_level_labels,
    reference_model = base$reference_model,
    reference_label = base$reference_label
  )
}

combined_pooled_table_base_to_latex <- function(combined, model_level_labels, reference_label = NULL) {

  library(tidyverse)

  cols <- c(
    "OL_NoModel_estimate_fmt", "OL_Main_estimate_fmt",
    "HS_NoModel_estimate_fmt", "HS_Main_estimate_fmt"
  )

  row_fmt <- function(term) {
    i <- match(term, combined$term_clean)
    vals <- unlist(combined[i, cols])
    vals <- ifelse(is.na(vals), "--", vals)
    paste(c(term, vals), collapse = " & ")
  }

  # suffix (e.g. "vs. GPT-4o mini") is appended outside \textbf so it reads as
  # a plain-text qualifier on the bolded section label, not part of the heading itself
  subheader <- function(label, suffix = NULL) {
    paste0(
      "\\multicolumn{5}{l}{\\textbf{", label, "}",
      if (!is.null(suffix)) paste0(" (", suffix, ")") else "",
      "} \\\\"
    )
  }

  task_terms <- c(
    "Joint: OL first (vs. Separate Labeling)",
    "Joint: HS first (vs. Separate Labeling)"
  )
  design_terms <- c("With Confidence (vs. Without)", "Batch Prompt (vs. Indiv. Prompts)")
  model_terms <- model_level_labels
  random_effect_terms <- c("$\\tau_{00}$\\textsubscript{tweet\\_id}")
  model_stats_terms <- c("ICC", "Marginal $R^2$", "Conditional $R^2$", "AIC", "BIC")

  model_suffix <- if (!is.null(reference_label)) paste0("vs. ", reference_label) else NULL

  body <- c(
    paste0(row_fmt("(Intercept)"), " \\\\"),
    subheader("Task Structure Effects"),
    paste0(map_chr(task_terms, row_fmt), " \\\\"),
    paste0(map_chr(design_terms, row_fmt), " \\\\"),
    subheader("Model Fixed Effects", model_suffix),
    paste0(map_chr(model_terms, row_fmt), " \\\\"),
    subheader("Random Effects"),
    paste0(map_chr(random_effect_terms, row_fmt), " \\\\"),
    "\\midrule",
    paste0(map_chr(model_stats_terms, row_fmt), " \\\\")
  )

  paste(body, collapse = "\n")
}

combined_pooled_table_base_to_standalone_tex <- function(combined, model_level_labels, reference_label = NULL) {

  body_rows <- combined_pooled_table_base_to_latex(combined, model_level_labels, reference_label)

  header <- paste(
    c("Term", "OL Task", "OL Models", "HS Task", "HS Models"),
    collapse = " & "
  )

  n_tweet <- trimws(combined$OL_Main_estimate_fmt[combined$term_clean == "$N$\\textsubscript{tweet\\_id}"])
  obs     <- trimws(combined$OL_Main_estimate_fmt[combined$term_clean == "Observations"])

  note <- c(
    "\\smallskip",
    "\\begin{minipage}{\\textwidth}",
    "\\raggedright",
    paste0(
      "\\footnotesize \\textit{Note}: $N$\\textsubscript{tweet\\_id} = ", n_tweet,
      "; Observations = ", obs, ". *** $p<.001$, ** $p<.01$, * $p<.05$. ",
      "(Intercept) is Separate Labeling at the reference batch/confidence levels (and reference model, in the Models columns); Task pools across models. ",
      "Model $\\times$ Task and Model $\\times$ Batch/Confidence interactions are not shown here."
    ),
    "\\end{minipage}"
  )

  c(
    "\\documentclass{article}",
    "\\usepackage{booktabs}",
    "\\usepackage{amsmath}",
    "\\usepackage{graphicx}",
    "\\begin{document}",
    "\\begin{table}",
    "\\centering",
    "\\resizebox{\\textwidth}{!}{%",
    "\\begin{tabular}{lcccc}",
    "\\toprule",
    paste0(header, " \\\\"),
    "\\midrule",
    body_rows,
    "\\bottomrule",
    "\\end{tabular}%",
    "}",
    note,
    "\\end{table}",
    "\\end{document}"
  )
}

# thin wrapper kept for the existing gaussian/LPM call site (main.Rmd's pooled-lpm-fit
# chunk), which isn't affected by the logit crash and doesn't need to be split.
run_pooled_analysis <- function(
    model_dfs,
    model_labels,
    reference_model = "gpt4o_mini",
    family = c("gaussian", "binomial"),
    n_cores = min(4, max(1, parallel::detectCores() - 1)),
    glmer_optCtrl = list(maxit = 200, factr = 1e8)
) {
  family <- match.arg(family)

  base <- fit_pooled_base(model_dfs, model_labels, reference_model, family, n_cores, glmer_optCtrl)
  interaction <- fit_pooled_interaction(base)
  design_interaction <- fit_pooled_design_interaction(base)
  random_slopes <- if (family == "gaussian") fit_pooled_random_slopes(base) else NULL

  combine_pooled_results(base, interaction, design_interaction, random_slopes)
}
combined_pooled_table_to_latex <- function(combined, model_level_labels, reference_label = NULL) {

  library(tidyverse)

  cols <- c(
    "OL_NoModel_estimate_fmt", "OL_Main_estimate_fmt", "OL_Interaction_estimate_fmt", "OL_DesignInteraction_estimate_fmt",
    "HS_NoModel_estimate_fmt", "HS_Main_estimate_fmt", "HS_Interaction_estimate_fmt", "HS_DesignInteraction_estimate_fmt"
  )

  row_fmt <- function(term) {
    i <- match(term, combined$term_clean)
    vals <- unlist(combined[i, cols])
    vals <- ifelse(is.na(vals), "--", vals)
    paste(c(term, vals), collapse = " & ")
  }

  subheader <- function(label, suffix = NULL) {
    paste0(
      "\\multicolumn{9}{l}{\\textbf{", label, "}",
      if (!is.null(suffix)) paste0(" (", suffix, ")") else "",
      "} \\\\"
    )
  }

  task_terms <- c(
    "Joint: OL first (vs. Separate Labeling)",
    "Joint: HS first (vs. Separate Labeling)"
  )
  design_terms <- c("With Confidence (vs. Without)", "Batch Prompt (vs. Indiv. Prompts)")
  model_terms     <- model_level_labels
  model_int_terms <- paste0(model_level_labels, " x Asked First")
  model_di_batch_terms <- paste0(model_level_labels, " x Batch Prompt")
  model_di_conf_terms  <- paste0(model_level_labels, " x With Confidence")
  random_effect_terms <- c("$\\tau_{00}$\\textsubscript{tweet\\_id}")
  model_stats_terms   <- c("ICC", "Marginal $R^2$", "Conditional $R^2$", "AIC", "BIC")

  model_suffix <- if (!is.null(reference_label)) paste0("vs. ", reference_label) else NULL

  body <- c(
    paste0(row_fmt("(Intercept)"), " \\\\"),
    subheader("Task Structure Effects"),
    paste0(map_chr(task_terms, row_fmt), " \\\\"),
    paste0(map_chr(design_terms, row_fmt), " \\\\"),
    subheader("Model Fixed Effects", model_suffix),
    paste0(map_chr(model_terms, row_fmt), " \\\\"),
    subheader("Model $\\times$ Task Structure Interaction"),
    paste0(map_chr(model_int_terms, row_fmt), " \\\\"),
    subheader("Model $\\times$ Batch/Confidence Interaction"),
    paste0(map_chr(model_di_batch_terms, row_fmt), " \\\\"),
    paste0(map_chr(model_di_conf_terms, row_fmt), " \\\\"),
    subheader("Random Effects"),
    paste0(map_chr(random_effect_terms, row_fmt), " \\\\"),
    "\\midrule",
    paste0(map_chr(model_stats_terms, row_fmt), " \\\\")
  )

  paste(body, collapse = "\n")
}

combined_pooled_table_to_standalone_tex <- function(combined, model_level_labels, reference_label = NULL) {

  body_rows <- combined_pooled_table_to_latex(combined, model_level_labels, reference_label)

  header <- paste(
    c(
      "Term",
      "OL Task", "OL Models", "OL Task Interaction", "OL Design Interaction",
      "HS Task", "HS Models", "HS Task Interaction", "HS Design Interaction"
    ),
    collapse = " & "
  )

  n_tweet <- trimws(combined$OL_Main_estimate_fmt[combined$term_clean == "$N$\\textsubscript{tweet\\_id}"])
  obs     <- trimws(combined$OL_Main_estimate_fmt[combined$term_clean == "Observations"])

  note <- c(
    "\\smallskip",
    "\\begin{minipage}{\\textwidth}",
    "\\raggedright",
    paste0(
      "\\footnotesize \\textit{Note}: $N$\\textsubscript{tweet\\_id} = ", n_tweet,
      "; Observations = ", obs, ". *** $p<.001$, ** $p<.01$, * $p<.05$. ",
      "(Intercept) is Separate Labeling at the reference batch/confidence levels (and reference model, in the Models/Interaction columns); Task pools across models."
    ),
    "\\end{minipage}"
  )

  c(
    "\\documentclass{article}",
    "\\usepackage{booktabs}",
    "\\usepackage{amsmath}",
    "\\usepackage{graphicx}",
    "\\begin{document}",
    "\\begin{table}",
    "\\centering",
    "\\resizebox{\\textwidth}{!}{%",
    "\\begin{tabular}{lcccccccc}",
    "\\toprule",
    paste0(header, " \\\\"),
    "\\midrule",
    body_rows,
    "\\bottomrule",
    "\\end{tabular}%",
    "}",
    note,
    "\\end{table}",
    "\\end{document}"
  )
}

combined_pooled_table_randomslopes_to_latex <- function(combined, model_level_labels, reference_label = NULL) {

  library(tidyverse)

  cols <- c(
    "OL_NoModel_estimate_fmt", "OL_Main_estimate_fmt", "OL_RandomSlopes_estimate_fmt",
    "HS_NoModel_estimate_fmt", "HS_Main_estimate_fmt", "HS_RandomSlopes_estimate_fmt"
  )

  row_fmt <- function(term) {
    i <- match(term, combined$term_clean)
    vals <- unlist(combined[i, cols])
    vals <- ifelse(is.na(vals), "--", vals)
    paste(c(term, vals), collapse = " & ")
  }

  subheader <- function(label, suffix = NULL) {
    paste0(
      "\\multicolumn{7}{l}{\\textbf{", label, "}",
      if (!is.null(suffix)) paste0(" (", suffix, ")") else "",
      "} \\\\"
    )
  }

  task_terms <- c(
    "Joint: OL first (vs. Separate Labeling)",
    "Joint: HS first (vs. Separate Labeling)"
  )
  design_terms   <- c("With Confidence (vs. Without)", "Batch Prompt (vs. Indiv. Prompts)")
  model_terms    <- model_level_labels
  variance_terms <- combined$term_clean[str_starts(combined$term_clean, "Var: ")]
  random_effect_terms <- c("$\\tau_{00}$\\textsubscript{tweet\\_id}")
  model_stats_terms   <- c("ICC", "Marginal $R^2$", "Conditional $R^2$", "AIC", "BIC")
  lrt_terms <- combined$term_clean[str_starts(combined$term_clean, "LRT")]

  model_suffix <- if (!is.null(reference_label)) paste0("vs. ", reference_label) else NULL

  body <- c(
    paste0(row_fmt("(Intercept)"), " \\\\"),
    subheader("Task Structure Effects"),
    paste0(map_chr(task_terms, row_fmt), " \\\\"),
    paste0(map_chr(design_terms, row_fmt), " \\\\"),
    subheader("Model Fixed Effects", model_suffix),
    paste0(map_chr(model_terms, row_fmt), " \\\\"),
    subheader("Cross-Model Variance Components"),
    paste0(map_chr(variance_terms, row_fmt), " \\\\"),
    subheader("Random Effects"),
    paste0(map_chr(random_effect_terms, row_fmt), " \\\\"),
    "\\midrule",
    paste0(map_chr(model_stats_terms, row_fmt), " \\\\"),
    paste0(map_chr(lrt_terms, row_fmt), " \\\\")
  )

  paste(body, collapse = "\n")
}

combined_pooled_table_randomslopes_to_standalone_tex <- function(combined, model_level_labels, reference_label = NULL) {

  body_rows <- combined_pooled_table_randomslopes_to_latex(combined, model_level_labels, reference_label)

  header <- paste(
    c("Term", "OL Task", "OL Models", "OL Random Slopes", "HS Task", "HS Models", "HS Random Slopes"),
    collapse = " & "
  )

  n_tweet <- trimws(combined$OL_Main_estimate_fmt[combined$term_clean == "$N$\\textsubscript{tweet\\_id}"])
  obs     <- trimws(combined$OL_Main_estimate_fmt[combined$term_clean == "Observations"])

  note <- c(
    "\\smallskip",
    "\\begin{minipage}{\\textwidth}",
    "\\raggedright",
    paste0(
      "\\footnotesize \\textit{Note}: $N$\\textsubscript{tweet\\_id} = ", n_tweet,
      "; Observations = ", obs, ". *** $p<.001$, ** $p<.01$, * $p<.05$. ",
      "(Intercept) is Separate Labeling at the reference batch/confidence levels (and reference model, in the Models/Random Slopes columns); Task pools across models. ",
      "Random Slopes adds uncorrelated per-model random slopes for task structure, batch, and confidence to the Models specification; LRT tests it against Models (REML)."
    ),
    "\\end{minipage}"
  )

  c(
    "\\documentclass{article}",
    "\\usepackage{booktabs}",
    "\\usepackage{amsmath}",
    "\\usepackage{graphicx}",
    "\\begin{document}",
    "\\begin{table}",
    "\\centering",
    "\\resizebox{\\textwidth}{!}{%",
    "\\begin{tabular}{lcccccc}",
    "\\toprule",
    paste0(header, " \\\\"),
    "\\midrule",
    body_rows,
    "\\bottomrule",
    "\\end{tabular}%",
    "}",
    note,
    "\\end{table}",
    "\\end{document}"
  )
}

# combined_paper_table now has the same 4-spec shape (NoModel/Main/Interaction/
# DesignInteraction) for both families, so these just delegate to the shared
# implementation - kept as separate names since main.Rmd calls them for the logit table.
combined_pooled_table_logit_to_latex <- function(combined, model_level_labels, reference_label = NULL) {
  combined_pooled_table_to_latex(combined, model_level_labels, reference_label)
}

combined_pooled_table_logit_to_standalone_tex <- function(combined, model_level_labels, reference_label = NULL) {

  body_rows <- combined_pooled_table_logit_to_latex(combined, model_level_labels, reference_label)

  header <- paste(
    c(
      "Term",
      "OL Task", "OL Models", "OL Task Interaction", "OL Design Interaction",
      "HS Task", "HS Models", "HS Task Interaction", "HS Design Interaction"
    ),
    collapse = " & "
  )

  n_tweet <- trimws(combined$OL_Main_estimate_fmt[combined$term_clean == "$N$\\textsubscript{tweet\\_id}"])
  obs     <- trimws(combined$OL_Main_estimate_fmt[combined$term_clean == "Observations"])

  note <- c(
    "\\smallskip",
    "\\begin{minipage}{\\textwidth}",
    "\\raggedright",
    paste0(
      "\\footnotesize \\textit{Note}: $N$\\textsubscript{tweet\\_id} = ", n_tweet,
      "; Observations = ", obs, ". *** $p<.001$, ** $p<.01$, * $p<.05$. ",
      "(Intercept) is Separate Labeling at the reference batch/confidence levels (and reference model, in the Models/Interaction columns); Task pools across models."
    ),
    "\\end{minipage}"
  )

  c(
    "\\documentclass{article}",
    "\\usepackage{booktabs}",
    "\\usepackage{amsmath}",
    "\\usepackage{graphicx}",
    "\\begin{document}",
    "\\begin{table}",
    "\\centering",
    "\\resizebox{\\textwidth}{!}{%",
    "\\begin{tabular}{lcccccccc}",
    "\\toprule",
    paste0(header, " \\\\"),
    "\\midrule",
    body_rows,
    "\\bottomrule",
    "\\end{tabular}%",
    "}",
    note,
    "\\end{table}",
    "\\end{document}"
  )
}
