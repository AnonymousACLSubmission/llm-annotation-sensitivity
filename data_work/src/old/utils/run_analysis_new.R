################################################################################
### run_full_analysis()
###
### Per-model prevalence engine behind Figure 2. Reshapes one model's label
### data to long format and returns condition-level OL/HS prevalences; also
### (re)builds the global `ref_lines` used by main.R's plot.
################################################################################

run_full_analysis <- function(combined_df, model_name, tweets_full, final_summary) {

  library(tidyverse)
  library(stringr)

  ### Wide to long: one row per (tweet, condition, responder)
  df_long <- combined_df %>%
    select(batch_id, tweet_in_batch, tweet_id, tweet, condition, shot, starts_with("R")) %>%
    pivot_longer(
      cols      = starts_with("R"),
      names_to  = c("responder", ".value"),
      names_sep = "_"
    ) %>%
    mutate(
      OL = as.integer(str_detect(label, "(?<!N)OL")),
      HS = as.integer(str_detect(label, "(?<!N)HS"))
    )

  ### Derive condition structure: task (A/B/C.OL/C.HS), batch, conf
  df_long <- df_long %>%
    mutate(
      condition_base = case_when(
        str_detect(condition, "^C\\.OL") ~ "C.OL",
        str_detect(condition, "^C\\.HS") ~ "C.HS",
        str_detect(condition, "^A")      ~ "A",
        str_detect(condition, "^B")      ~ "B",
        TRUE                             ~ NA_character_
      ),
      condition_label = case_when(
        condition_base == "A"                    ~ "Joint,\nOL first",
        condition_base == "B"                     ~ "Joint,\nHS first",
        condition_base %in% c("C.OL", "C.HS")      ~ "Separate",
        TRUE                                        ~ condition_base
      ),
      condition_label = factor(
        condition_label,
        levels = c("Joint,\nOL first", "Joint,\nHS first", "Separate")
      ),
      group_type = case_when(
        str_detect(condition, "_batch_conf$") ~ "batch_conf",
        str_detect(condition, "_conf$")       ~ "conf",
        str_detect(condition, "_batch$")      ~ "batch",
        TRUE                                   ~ "base"
      ),
      group_type = factor(group_type, levels = c("base", "conf", "batch", "batch_conf"))
    )

  ### OL / HS prevalence by condition (drop the mismatched half: C.HS
  ### rows have no valid OL judgement in the "separate" design, and vice
  ### versa for C.OL rows and HS)
  df_OL <- df_long %>%
    filter(condition_base != "C.HS") %>%
    transmute(condition_label, group_type, Measure = "OL", Value = OL)

  df_HS <- df_long %>%
    filter(condition_base != "C.OL") %>%
    transmute(condition_label, group_type, Measure = "HS", Value = HS)

  prevalence_data <- bind_rows(df_OL, df_HS) %>%
    group_by(condition_label, group_type, Measure) %>%
    summarise(Value = mean(Value, na.rm = TRUE), .groups = "drop") %>%
    mutate(
      Value_Pct  = Value * 100,
      model_name = model_name,
      Measure    = factor(Measure, levels = c("OL", "HS"))
    )

  ### Human reference lines -- exposed globally since main.R's plot
  ### references a bare `ref_lines` object
  overall_OL <- final_summary %>% filter(version == "overall") %>% pull(OL)
  overall_HS <- final_summary %>% filter(version == "overall") %>% pull(HS)

  ref_lines <<- data.frame(
    Measure   = factor(c("OL", "HS"), levels = c("OL", "HS")),
    intercept = c(overall_OL * 100, overall_HS * 100)
  )

  list(
    df_long         = df_long,
    prevalence_data = prevalence_data
  )
}
