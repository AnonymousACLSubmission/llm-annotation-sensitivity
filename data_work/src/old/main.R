################################################################################
### Setup
################################################################################

library(tidyverse)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
rm(list = ls())

source("run_analysis_new.R")
source("run_table_wide.R")
source("run_pooled_analysis.R")

################################################################################
### Load Kern Data
################################################################################

folder_path <- "Tweets_CK"

train_df <- read_csv(file.path(folder_path, "full_train_s.csv"))
test_df  <- read_csv(file.path(folder_path, "full_test_s.csv"))

test_df <- test_df %>% select(all_of(names(train_df)))

tweets_full <- bind_rows(
  train_df %>% mutate(original_split = "train"),
  test_df  %>% mutate(original_split = "test")
) %>%
  rename(
    tweet_id = `tweet.id`,
    hate_speech = `hate.speech`,
    offensive_language = `offensive.language`
  )

summary <- tweets_full %>%
  summarise(
    OL = mean(offensive_language, na.rm = TRUE),
    HS = mean(hate_speech, na.rm = TRUE)
  )

final_summary <- tibble(
  version = "overall",
  OL = summary$OL,
  HS = summary$HS
)

################################################################################
### Load Models
################################################################################

load_model <- function(path, model_name) {
  files <- list.files(path, full.names = TRUE)

  map_df(files, read_csv) %>%
    arrange(batch_id, tweet_in_batch) %>%
    mutate(
      shot = "few-shot",
      model = model_name
    )
}

gpt4    <- load_model("Data_Collection/Models/GPT4o_mini",     "gpt4o_mini")
gpt54   <- load_model("Data_Collection/Models/GPT5.4",         "gpt5.4")
llama4  <- load_model("Data_Collection/Models/Llama4",         "llama4")
mistral <- load_model("Data_Collection/Models/MistralLarge3",  "mistral")

llama31_70b      <- load_model("Data_Collection/Models/Llama3.1_70B_new",    "llama31_70b")
llama31_8b       <- load_model("Data_Collection/Models/Llama3.1_8B_new",    "llama31_8b")
mistral_medium35 <- load_model("Data_Collection/Models/MistralMedium3.5_new", "mistral_medium35")

model_keys <- c(
  "gpt4o_mini", "gpt5.4", "mistral", "mistral_medium35",
  "llama31_8b", "llama31_70b", "llama4"
)

model_labels <- c(
  gpt4o_mini       = "GPT-4o mini",
  gpt5.4           = "GPT-5.4",
  mistral          = "Mistral Large 3",
  mistral_medium35 = "Mistral Medium 3.5",
  llama31_8b       = "Llama 3.1 8B",
  llama31_70b      = "Llama 3.1 70B",
  llama4           = "Llama 4"
)

model_dfs <- list(
  gpt4o_mini       = gpt4,
  gpt5.4           = gpt54,
  mistral          = mistral,
  mistral_medium35 = mistral_medium35,
  llama31_8b       = llama31_8b,
  llama31_70b      = llama31_70b,
  llama4           = llama4
)

################################################################################
### Run Prevalence Analyses
################################################################################

prevalence_results <- imap(model_dfs, function(df, key) {
  run_full_analysis(
    combined_df   = df,
    model_name    = key,
    tweets_full   = tweets_full,
    final_summary = final_summary
  )
})

df_prevalence_all_models <- map_dfr(prevalence_results, "prevalence_data") %>%
  mutate(
    model = factor(
      model_name,
      levels = model_keys,
      labels = model_labels[model_keys]
    )
  )

combined_prevalence_plot <- ggplot(
  df_prevalence_all_models,
  aes(x = condition_label, y = Value_Pct, color = group_type)
) +

  geom_hline(
    data = ref_lines,
    aes(
      yintercept = intercept,
      linetype = "Reference"
    ),
    color = "gray50",
    linewidth = 0.35,
    show.legend = TRUE
  ) +

  geom_point(
    position = position_dodge(width = 0.55),
    size = 1.6
  ) +

  facet_grid(
    model ~ Measure,
    scales = "fixed",
    labeller = labeller(
      Measure = c(OL = "OL", HS = "HS")
    )
  ) +

  scale_color_manual(
    name = "Condition",
    values = c(
      "base"       = "#443983",
      "conf"       = "#31688e",
      "batch"      = "#35b779",
      "batch_conf" = "#90d743"
    ),
    labels = c(
      "base"       = "Base",
      "conf"       = "+ Conf.",
      "batch"      = "Batch",
      "batch_conf" = "Batch + Conf."
    )
  ) +

  scale_linetype_manual(
    name = NULL,
    values = c("Reference" = "dashed")
  ) +

  scale_y_continuous(
    limits = c(0, 100),
    breaks = c(0, 50, 100),
    expand = expansion(mult = c(0.03, 0.05)),
    labels = function(x) paste0(x, "%")
  ) +

  labs(
    x = "Task Structure",
    y = "Prevalence"
  ) +

  guides(
    color = guide_legend(order = 1, nrow = 2),
    linetype = guide_legend(order = 2)
  ) +

  theme_bw(base_size = 8) +
  theme(
    strip.background = element_blank(),
    strip.text = element_text(size = 7.5, face = "bold"),

    legend.position = "bottom",
    legend.box = "vertical",
    legend.text = element_text(size = 6.5),
    legend.title = element_text(size = 7),
    legend.key.size = unit(0.3, "cm"),

    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_blank(),
    panel.grid.major.y = element_line(linewidth = 0.2, color = "gray85"),
    panel.border = element_rect(color = "gray75", linewidth = 0.3),

    axis.ticks = element_blank(),
    axis.text.x = element_text(size = 6.5, angle = 0, vjust = 1),
    axis.text.y = element_text(size = 6.5),

    panel.spacing = unit(0.3, "lines"),

    plot.margin = margin(t = 2, r = 2, b = 12, l = 2)
  )

ggsave(
  "Plots/combined_prevalences_all_models_acl_halfpage.pdf",
  combined_prevalence_plot,
  device = cairo_pdf,
  width = 8.8,
  height = 24,
  units = "cm"
)

################################################################################
### RUN INDIVIDUAL LPM MODELS
################################################################################

source("run_inference_analysis_v2.R")

lpm_results <- imap(model_dfs, function(df, key) {
  run_inference_analysis(
    combined_df = df,
    model_name  = key
  )
})

################################################################################
### TABLE -- Wide, all 7 models
################################################################################

table1_wide <- build_wide_table(
  lpm_results  = lpm_results,
  model_order  = model_keys,
  model_labels = model_labels
)

write_csv(table1_wide, "Plots/model_lpm_tables/table1_wide_7models.csv")

table1_wide_latex <- wide_table_to_latex(table1_wide, model_labels[model_keys])
writeLines(table1_wide_latex, "Plots/model_lpm_tables/table1_wide_7models_latex.txt")

################################################################################
### TABLE -- Pooled model with "Model:" dummy terms
################################################################################

pooled_results <- run_pooled_analysis(
  model_dfs    = model_dfs,
  model_labels = model_labels
)

pooled_dir <- "Plots/combined_inference"
dir.create(pooled_dir, recursive = TRUE, showWarnings = FALSE)

write_csv(pooled_results$fixed_table_full, file.path(pooled_dir, "combined_LPM_coefficients_long.csv"))
write_csv(pooled_results$fixed_table, file.path(pooled_dir, "combined_LPM_model_effects_coefficients.csv"))
write_csv(pooled_results$paper_table, file.path(pooled_dir, "combined_LPM_paper_table.csv"))

pooled_latex <- paper_table_to_latex(pooled_results$paper_table)
writeLines(pooled_latex, file.path(pooled_dir, "combined_LPM_latex_rows.txt"))
