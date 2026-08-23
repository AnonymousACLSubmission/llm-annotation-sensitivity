run_inference_analysis <- function(
    combined_df,
    model_name,
    output_dir = "Plots/model_lpm_tables"
) {
  
  library(tidyverse)
  library(stringr)
  library(lme4)
  library(lmerTest)
  library(broom.mixed)
  library(performance)
  library(readr)
  
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  
  cat("Running individual LPM inference for:", model_name, "\n")
  
  ################################################################################
  ### Preprocess Labels
  ################################################################################
  
  df_long <- combined_df %>%
    select(
      batch_id,
      tweet_in_batch,
      tweet_id,
      tweet,
      condition,
      shot,
      starts_with("R")
    ) %>%
    pivot_longer(
      cols = starts_with("R"),
      names_to = c("responder", ".value"),
      names_sep = "_"
    ) %>%
    mutate(
      OL = as.integer(str_detect(label, "(?<!N)OL")),
      HS = as.integer(str_detect(label, "(?<!N)HS")),
      batch = ifelse(grepl("batch", condition), 1, 0),
      conf  = ifelse(grepl("conf", condition), 1, 0),
      task = case_when(
        grepl("C\\.OL", condition) ~ "C.OL",
        grepl("C\\.HS", condition) ~ "C.HS",
        grepl("^A", condition)     ~ "A",
        grepl("^B", condition)     ~ "B",
        TRUE                       ~ NA_character_
      )
    )
  
  ################################################################################
  ### OL Dataset
  ################################################################################
  
  df_OL <- df_long %>%
    filter(!grepl("C\\.HS", condition)) %>%
    mutate(
      task = factor(
        task,
        levels = c("A", "B", "C.OL"),
        labels = c("Joint: OL first", "Joint: HS first", "Separate Labeling")
      ),
      batch = factor(batch),
      conf  = factor(conf)
    )
  
  contrasts(df_OL$task) <- contr.sum(levels(df_OL$task))
  
  ################################################################################
  ### HS Dataset
  ################################################################################
  
  df_HS <- df_long %>%
    filter(!grepl("C\\.OL", condition)) %>%
    mutate(
      task = factor(
        task,
        levels = c("A", "B", "C.HS"),
        labels = c("Joint: OL first", "Joint: HS first", "Separate Labeling")
      ),
      batch = factor(batch),
      conf  = factor(conf)
    )
  
  contrasts(df_HS$task) <- contr.sum(levels(df_HS$task))
  
  ################################################################################
  ### Fit LPMs
  ################################################################################
  
  model_OL_linprob <- lmer(
    OL ~ 0 + task + batch + conf + (1 | tweet_id),
    data = df_OL
  )
  
  model_HS_linprob <- lmer(
    HS ~ 0 + task + batch + conf + (1 | tweet_id),
    data = df_HS
  )
  
  ################################################################################
  ### Helper: extract fixed effects
  ################################################################################
  
  extract_fixed <- function(model, outcome) {
    
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
          term == "taskJoint: OL first" ~ "Joint: OL first",
          term == "taskJoint: HS first" ~ "Joint: HS first",
          term == "taskSeparate Labeling" ~ "Separate Labeling",
          term == "batch1" ~ "Batch Prompt (vs. Indiv. Prompts)",
          term == "conf1" ~ "With Confidence (vs. Without)",
          TRUE ~ term
        )
      ) %>%
      select(outcome, term_clean, estimate, std.error, statistic, p.value, stars)
  }
  
  fixed_table <- bind_rows(
    extract_fixed(model_OL_linprob, "OL"),
    extract_fixed(model_HS_linprob, "HS")
  )
  
  ################################################################################
  ### Helper: extract random effects / model statistics
  ################################################################################
  
  extract_model_stats <- function(model, outcome, n_tweets) {
    
    vc <- as.data.frame(VarCorr(model))
    
    sigma2 <- sigma(model)^2
    
    tau00 <- vc %>%
      filter(grp == "tweet_id") %>%
      pull(vcov)
    
    icc <- tau00 / (tau00 + sigma2)
    
    r2 <- performance::r2_nakagawa(model)
    
    tibble(
      outcome = outcome,
      sigma2 = sigma2,
      tau00_tweet_id = tau00,
      ICC = icc,
      N_tweet_id = n_tweets,
      Observations = nobs(model),
      Marginal_R2 = as.numeric(r2$R2_marginal),
      Conditional_R2 = as.numeric(r2$R2_conditional)
    )
  }
  
  stats_table <- bind_rows(
    extract_model_stats(
      model_OL_linprob,
      "OL",
      n_distinct(df_OL$tweet_id)
    ),
    extract_model_stats(
      model_HS_linprob,
      "HS",
      n_distinct(df_HS$tweet_id)
    )
  )
  
  ################################################################################
  ### Paper-style wide coefficient table
  ################################################################################
  
  coef_wide <- fixed_table %>%
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
    select(
      term_clean,
      OL_estimate_fmt,
      OL_se_fmt,
      HS_estimate_fmt,
      HS_se_fmt
    )
  
  stats_wide <- stats_table %>%
    pivot_longer(
      cols = c(
        sigma2,
        tau00_tweet_id,
        ICC,
        N_tweet_id,
        Observations,
        Marginal_R2,
        Conditional_R2
      ),
      names_to = "term_clean",
      values_to = "value"
    ) %>%
    mutate(
      value_fmt = case_when(
        term_clean %in% c("N_tweet_id", "Observations") ~ format(round(value), big.mark = ",", scientific = FALSE),
        TRUE ~ sprintf("%.3f", value)
      )
    ) %>%
    select(outcome, term_clean, value_fmt) %>%
    pivot_wider(
      names_from = outcome,
      values_from = value_fmt,
      names_glue = "{outcome}_estimate_fmt"
    ) %>%
    mutate(
      OL_se_fmt = "--",
      HS_se_fmt = "--"
    ) %>%
    select(
      term_clean,
      OL_estimate_fmt,
      OL_se_fmt,
      HS_estimate_fmt,
      HS_se_fmt
    ) %>%
    mutate(
      term_clean = recode(
        term_clean,
        "sigma2" = "$\\sigma^2$",
        "tau00_tweet_id" = "$\\tau_{00}$\\textsubscript{tweet\\_id}",
        "N_tweet_id" = "$N$\\textsubscript{tweet\\_id}",
        "Marginal_R2" = "Marginal $R^2$",
        "Conditional_R2" = "Conditional $R^2$"
      )
    )
  
  paper_table <- bind_rows(
    coef_wide,
    stats_wide
  )
  
  ################################################################################
  ### LaTeX table body
  ################################################################################
  
  latex_rows <- paper_table %>%
    mutate(
      latex_row = paste0(
        term_clean, " & ",
        OL_estimate_fmt, " & ", OL_se_fmt, " & ",
        HS_estimate_fmt, " & ", HS_se_fmt, " \\\\"
      )
    ) %>%
    pull(latex_row)
  
  latex_table_body <- paste(latex_rows, collapse = "\n")
  
  ################################################################################
  ### Save outputs
  ################################################################################
  
  write_csv(
    fixed_table,
    file.path(output_dir, paste0(model_name, "_LPM_coefficients_long.csv"))
  )
  
  write_csv(
    stats_table,
    file.path(output_dir, paste0(model_name, "_LPM_model_stats.csv"))
  )
  
  write_csv(
    paper_table,
    file.path(output_dir, paste0(model_name, "_LPM_paper_table.csv"))
  )
  
  writeLines(
    latex_table_body,
    file.path(output_dir, paste0(model_name, "_LPM_latex_rows.txt"))
  )
  
  cat("Finished individual LPM inference for:", model_name, "\n")
  
  return(list(
    model_OL_linprob = model_OL_linprob,
    model_HS_linprob = model_HS_linprob,
    df_long = df_long,
    df_OL = df_OL,
    df_HS = df_HS,
    fixed_table = fixed_table,
    stats_table = stats_table,
    paper_table = paper_table,
    latex_table_body = latex_table_body
  ))
}