################################################################################
### Pre-Req
################################################################################

### load packages
library(tidyverse)
library(dplyr)
library(tidyr)
library(ggplot2)
library(stringr)
library(DescTools)  
library(irr)  
library(tidyverse)   
library(stringr)
library(patchwork)   
library(RColorBrewer)
library(data.table)
library(purrr)
library(lme4)

### Set WD and clear environment
setwd(dirname(rstudioapi::getActiveDocumentContext()$path)) # wd = source loc
# setwd("~/LRZ Sync+Share/Thomas Reiter/MasterThesis/MasterThesis")
rm(list = ls())


################################################################################
### Load the Kern Original Data
################################################################################

folder_path <- "Tweets_CK"
train_file  <- "full_train_s.csv"
test_file   <- "full_test_s.csv"

train_df <- read_csv(file.path(folder_path, train_file))
test_df  <- read_csv(file.path(folder_path, test_file))


# reorder test_df columns to match train_df
test_df <- test_df %>% select(all_of(names(train_df)))

# add the 'original_split' column
train_df <- train_df %>% mutate(original_split = "train")
test_df  <- test_df  %>% mutate(original_split = "test")

# combine
tweets_full <- bind_rows(train_df, test_df)

# consistent naming
tweets_full <- tweets_full %>%
  rename(
    tweet_id           = `tweet.id`,
    batch_tweet        = `batch.tweet`,
    hate_speech        = `hate.speech`,
    offensive_language = `offensive.language`
  )


################################################################################
### Reproduce the Kern et al. (2023) Table for Test Purposes 
################################################################################
summary <- tweets_full %>%
  group_by(version) %>%
  summarise(
    Annotations = n(),                     # count of tweets
    Annotators  = n_distinct(id),          # number of unique annotators
    OL          = mean(offensive_language, na.rm = T),
    HS          = mean(hate_speech, na.rm = T)
  ) %>%
  mutate(
    across(c(OL, HS), ~ round(.x, 3))      # round to 3 decimals
  )

total_annotations <- sum(summary$Annotations)
total_annotators  <- sum(summary$Annotators)
weighted_OL       <- sum(summary$OL * summary$Annotations) / total_annotations
weighted_HS       <- sum(summary$HS * summary$Annotations) / total_annotations

overall <- tibble(
  version     = "overall",
  Annotations = total_annotations,
  Annotators  = total_annotators,
  OL          = round(weighted_OL, 3),
  HS          = round(weighted_HS, 3)
)

final_summary <- bind_rows(summary, overall)
print(final_summary)


################################################################################
### Load and RBind the GPT Labelled Tweets
################################################################################

# zero shot (older and smaller sample size --> only used during initial expl.)
date_to_import_z <- "2025_04_10"
files_z <- list.files(
  path = "Data_Collection/zero-shot/",
  pattern = paste0("^", date_to_import_z),
  full.names = TRUE
)

combined_df_z <- files_z %>%
  map_df(read_csv, .id = "source") %>%     
  arrange(batch_id, tweet_in_batch) %>%
  mutate(shot = "zero-shot") %>%
  select(1:4, shot, everything())

combined_df_z$R1_score <- as.character(combined_df_z$R1_score)
combined_df_z$R2_score <- as.character(combined_df_z$R2_score)
combined_df_z$R3_score <- as.character(combined_df_z$R3_score)

# few shot (main sample)
date_to_import_f <- "2025_07_01"
files_f <- list.files(
  path = "Data_Collection/OL_NH/",
  pattern = paste0("^", date_to_import_f),
  full.names = TRUE
)

# Additional Models Gpt5.4
date_to_import_gpt54 <- c("2026_03_13", "2026_03_17")
files_f_gpt54 <- list.files(
  path = "Data_Collection/AdditionalModels/GPT5.4",
  # pattern = paste0("^", date_to_import_gpt54),
  full.names = TRUE
)

# Additional Models Llama4
files_f_llama4 <- list.files(
  path = "Data_Collection/AdditionalModels/Llama4",
  # pattern = paste0("^", date_to_import_gpt54),
  full.names = TRUE
)

# Additional Models MistralLarge3
files_f_mistral <- list.files(
  path = "Data_Collection/AdditionalModels/MistralLarge3",
  # pattern = paste0("^", date_to_import_gpt54),
  full.names = TRUE
)



combined_df_f_gptmini <- files_f %>%
  map_df(read_csv, .id = "source") %>%
  arrange(batch_id, tweet_in_batch) %>%
  mutate(shot = "few-shot") %>%
  select(1:4, shot, everything())

combined_df_f_gpt54 <- files_f_gpt54 %>%
  map_df(read_csv, .id = "source") %>%
  arrange(batch_id, tweet_in_batch) %>%
  mutate(shot = "few-shot") %>%
  select(1:4, shot, everything())

combined_df_f_llama4 <- files_f_llama4 %>%
  map_df(read_csv, .id = "source") %>%
  arrange(batch_id, tweet_in_batch) %>%
  mutate(shot = "few-shot") %>%
  select(1:4, shot, everything())

combined_df_f_mistral <- files_f_mistral %>%
  map_df(read_csv, .id = "source") %>%
  arrange(batch_id, tweet_in_batch) %>%
  mutate(shot = "few-shot") %>%
  select(1:4, shot, everything())


combined_df <- bind_rows(combined_df_z, combined_df_f)

n_tweets        <- combined_df %>% pull(tweet_id) %>% unique() %>% length()
n_tweet_batches <- combined_df %>% pull(batch_id)   %>% unique() %>% length()
n_conditions    <- combined_df %>% pull(condition)  %>% unique() %>% length()
cat("Unique Tweets: ", n_tweets,        "\n")
cat("No. Batches:   ", n_tweet_batches, "\n")
cat("Conditions:    ", n_conditions,    "\n")



################################################################################
### Preprocess GPT Labels
################################################################################
shared_cols <- c(
  "batch_id", "tweet_in_batch", "tweet_id", 
  "tweet", "condition", "shot"
)

# wide to long
df_long <- combined_df %>%
  select(batch_id, tweet_in_batch, tweet_id, tweet, condition, shot, starts_with("R")) %>%
  pivot_longer(
    cols      = starts_with("R"),
    names_to  = c("responder", ".value"),
    names_sep = "_"
  )

# OL / HS label cols
df_long <- df_long %>%
  mutate(
    OL = as.integer(str_detect(label, "(?<!N)OL")),
    HS = as.integer(str_detect(label, "(?<!N)HS"))
  )



################################################################################
### Compute and Compare Overall Prevalences
### Main Paper Plots (few-shot only)
################################################################################

tasks        <- c("A", "B", "C.OL", "C.HS")
modes        <- c("", "_conf", "_batch", "_batch_conf")
custom_order <- as.vector(outer(tasks, modes, paste0))
shot_order   <- c("zero-shot", "few-shot")

df_long <- df_long %>%
  mutate(
    condition = factor(condition, levels = custom_order),
    shot      = factor(shot,      levels = shot_order)
  )


################################################################################
### Plot Few-Shot Prevalences
################################################################################
overall_OL <- final_summary %>% 
  filter(version == "overall") %>% 
  pull(OL)
overall_HS <- final_summary %>% 
  filter(version == "overall") %>% 
  pull(HS)

shot_order  <- c("zero-shot", "few-shot")
group_order <- c("base", "conf", "batch", "batch_conf")

### custom order
tasks        <- c("A", "B", "C.OL", "C.HS")
modes        <- c("", "_conf", "_batch", "_batch_conf")
custom_order <- as.vector(outer(tasks, modes, paste0))

### build long format df for plot
df_plot <- df_long %>%
  # rename s.t. we have a base condition
  mutate(
    base_condition = word(condition, 1, sep = "_"),
    group_type = case_when(
      !str_detect(condition, "_")            ~ "base",
      str_detect(condition, "_batch_conf$")  ~ "batch_conf",
      str_detect(condition, "_conf$")        ~ "conf",
      str_detect(condition, "_batch$")       ~ "batch",
      TRUE                                   ~ "other"
    )
  ) %>%
  # wide to long for OL and HS
  pivot_longer(
    cols      = c(OL, HS),
    names_to  = "Measure",
    values_to = "Value"
  ) %>%
  # factor order for plots
  mutate(
    condition  = factor(condition,  levels = custom_order),
    group_type = factor(group_type, levels = group_order),
    shot       = factor(shot, levels = shot_order,
                        labels = c("Proportion\n(zero-shot)",
                                   "Proportion\n(few-shot)")),
    Measure    = factor(Measure, levels = c("OL","HS"))
  )


### Compute the mean prevalences 
df_prop <- df_plot %>%
  group_by(condition, group_type, shot, Measure) %>%
  summarise(
    Value = mean(Value, na.rm = TRUE),
    .groups = "drop")

overall_OL <- final_summary %>% filter(version == "overall") %>% pull(OL)
overall_HS <- final_summary %>% filter(version == "overall") %>% pull(HS)


##### combine C.OL and C.HS conditions
# remove the 'wrong rows'
id_out_1 <- which(grepl("C\\.OL", df_prop$condition) & df_prop$Measure == "HS")
id_out_2 <- which(grepl("C\\.HS", df_prop$condition) & df_prop$Measure == "OL")

df_prop <- df_prop[-c(id_out_1, id_out_2), ]
strip_pattern <- "\\.(?:OL|HS)"
df_prop$condition <- str_remove_all(df_prop$condition, strip_pattern)


####################
### Few-Shot Prevalences Dot Plot (1x2)

# 1. Filter for Few-Shot and Prepare Data
df_plot_fewshot <- df_prop %>%
  # Filter for few-shot only
  filter(shot == "Proportion\n(few-shot)") %>%
  mutate(
    # Clean condition names to base task letters (A, B, C)
    condition_base = str_remove(condition, "_.*"),
    # Map letters to descriptive labels
    condition_label = case_when(
      condition_base == "A" ~ "Joint,\nOL first",
      condition_base == "B" ~ "Joint,\nHS first",
      condition_base == "C" ~ "Separate",
      TRUE ~ condition_base
    ),
    condition_label = factor(condition_label, 
                             levels = c("Joint,\nOL first", "Joint,\nHS first", "Separate")),
    # Scale to 100 for percentage display
    Value_Pct = Value * 100
  )

# 2. Reference line data (few-shot context)
ref_lines <- data.frame(
  Measure = factor(c("OL", "HS"), levels = c("OL", "HS")),
  intercept = c(overall_OL * 100, overall_HS * 100)
)

# Plot
ggplot(df_plot_fewshot, aes(x = condition_label, y = Value_Pct, color = group_type)) +
  
  ## Reference lines (mapped so they appear in legend)
  geom_hline(
    data = ref_lines,
    aes(
      yintercept = intercept,
      linetype   = "Reference\n(Kern et al., 2023)"
    ),
    color = "gray50",
    linewidth = 0.7
  ) +
  
  ## Dot layer
  geom_point(
    position = position_dodge(width = 0.6),
    size = 3.5
  ) +
  
  ## Vertical facet layout
  facet_grid(
    Measure ~ .,
    scales   = "fixed",
    labeller = labeller(Measure = c(OL = "Measure: OL", HS = "Measure: HS"))
  ) +
  
  ## Condition color legend (FIRST)
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
      "conf"       = "+ Confidence",
      "batch"      = "Batch",
      "batch_conf" = "Batch +\nConfidence"
    )
  ) +
  
  ## Reference legend (SECOND)
  scale_linetype_manual(
    name   = NULL,
    values = c("Reference\n(Kern et al., 2023)" = "dashed")
  ) +
  
  ## Y-axis
  scale_y_continuous(
    limits = c(0, 100),
    breaks = seq(0, 100, 25),
    expand = expansion(mult = c(0.1, 0.1)),
    labels = function(x) paste0(x, "%")
  ) +
  
  labs(
    x = "Task Structure", 
    y = "Prevalence (%)"
  ) +
  
  ## Legend layout & order
  guides(
    color    = guide_legend(order = 1, nrow = 2),
    linetype = guide_legend(order = 2)
  ) +
  
  theme_bw(base_size = 12) +
  theme(
    strip.background   = element_blank(),
    strip.text         = element_text(size = 11, face = "bold"),
    
    legend.position    = "bottom",
    legend.box         = "vertical",
    
    panel.grid.minor   = element_blank(),
    panel.grid.major.x = element_blank(),
    panel.border       = element_rect(color = "lightgray"),
    axis.ticks         = element_blank(),
    axis.text.x        = element_text(size = 10),
    
    legend.margin     = margin(t = -5),
    legend.box.margin = margin(t = -5),
    plot.margin       = margin(5, 5, 0, 5)
  )



# save (vertical)
ggsave("Plots/prevalences_few_ABC_dots_2x1.pdf",
       last_plot(),
       device = cairo_pdf,
       width  = 9,
       height = 13,
       units  = "cm"
)



################################################################################
### Compute and Compare Overall Prevalences
### Appendix Plots (zero-shot and few-shot subset comparison)
################################################################################

### get the zero-shot tweet subset for appendix comparison
zero_shot_tweets <- combined_df %>%
  filter(shot == "zero-shot") %>%
  distinct(tweet_id) %>%
  pull(tweet_id)

df_long_200 <- df_long %>%
  filter(tweet_id %in% zero_shot_tweets) %>%
  droplevels()

# Sanity checks
df_long_200 %>%
  distinct(tweet_id, shot) %>%
  count(shot)

# overwrite df long for original plot with the above subset df_long
summary_tbl_200 <- df_long_200 %>%
  group_by(condition, shot) %>%
  summarize(
    OL = round(mean(OL, na.rm = TRUE), 2),
    HS = round(mean(HS, na.rm = TRUE), 2),
    .groups = "drop"
  ) %>%
  arrange(condition, shot)

summary_tbl_200



################################################################################
### Plot Prevalences
################################################################################
overall_OL <- final_summary %>% 
  filter(version == "overall") %>% 
  pull(OL)
overall_HS <- final_summary %>% 
  filter(version == "overall") %>% 
  pull(HS)

shot_order  <- c("zero-shot", "few-shot")
group_order <- c("base", "conf", "batch", "batch_conf")

### custom order
tasks        <- c("A", "B", "C.OL", "C.HS")
modes        <- c("", "_conf", "_batch", "_batch_conf")
custom_order <- as.vector(outer(tasks, modes, paste0))

### build long format df for plot
df_plot_200 <- df_long_200 %>%
  # rename s.t. we have a base condition
  mutate(
    base_condition = word(condition, 1, sep = "_"),
    group_type = case_when(
      !str_detect(condition, "_")            ~ "base",
      str_detect(condition, "_batch_conf$")  ~ "batch_conf",
      str_detect(condition, "_conf$")        ~ "conf",
      str_detect(condition, "_batch$")       ~ "batch",
      TRUE                                   ~ "other"
    )
  ) %>%
  # wide to long for OL and HS
  pivot_longer(
    cols      = c(OL, HS),
    names_to  = "Measure",
    values_to = "Value"
  ) %>%
  # factor order for plots
  mutate(
    condition  = factor(condition,  levels = custom_order),
    group_type = factor(group_type, levels = group_order),
    shot       = factor(shot, levels = shot_order,
                        labels = c("Proportion\n(zero-shot)",
                                   "Proportion\n(few-shot)")),
    Measure    = factor(Measure, levels = c("OL","HS"))
  )


### Compute the mean prevalences 
df_prop_200 <- df_plot_200 %>%
  group_by(condition, group_type, shot, Measure) %>%
  summarise(
    Value = mean(Value, na.rm = TRUE),
    .groups = "drop")

overall_OL <- final_summary %>% filter(version == "overall") %>% pull(OL)
overall_HS <- final_summary %>% filter(version == "overall") %>% pull(HS)


##### combine C.OL and C.HS conditions
# remove the 'wrong rows'
id_out_1 <- which(grepl("C\\.OL", df_prop_200$condition) & df_prop_200$Measure == "HS")
id_out_2 <- which(grepl("C\\.HS", df_prop_200$condition) & df_prop_200$Measure == "OL")

df_prop_200 <- df_prop_200[-c(id_out_1, id_out_2), ]
strip_pattern <- "\\.(?:OL|HS)"
df_prop_200$condition <- str_remove_all(df_prop_200$condition, strip_pattern)



# 1. Prepare Data with a combined Facet variable
df_plot_4x1_200 <- df_prop_200 %>%
  mutate(
    # Clean condition names
    condition_base = str_remove(condition, "_.*"),
    condition_label = case_when(
      condition_base == "A" ~ "Joint,\nOL first",
      condition_base == "B" ~ "Joint,\nHS first",
      condition_base == "C" ~ "Separate",
      TRUE ~ condition_base
    ),
    condition_label = factor(condition_label, 
                             levels = c("Joint,\nOL first", "Joint,\nHS first", "Separate")),
    
    # Clean shot names for the label
    shot_clean = case_when(
      str_detect(shot, "zero") ~ "Zero-Shot",
      str_detect(shot, "few")  ~ "Few-Shot"
    ),
    
    # CREATE THE COMBINED FACET: This ensures the exact 4x1 order you requested
    # Order: OL Zero -> HS Zero -> OL Few -> HS Few
    facet_group = paste(Measure, "|", shot_clean),
    facet_group = factor(facet_group, levels = c(
      "OL | Zero-Shot", 
      "HS | Zero-Shot", 
      "OL | Few-Shot", 
      "HS | Few-Shot"
    )),
    
    Value_Pct = Value * 100
  )

# 2. Reference line data (must match the new facet_group levels)
ref_lines_4x1_200 <- data.frame(
  facet_group = factor(c("OL | Zero-Shot", "HS | Zero-Shot", "OL | Few-Shot", "HS | Few-Shot"),
                       levels = c("OL | Zero-Shot", "HS | Zero-Shot", "OL | Few-Shot", "HS | Few-Shot")),
  intercept = c(overall_OL * 100, overall_HS * 100, overall_OL * 100, overall_HS * 100)
)

# 3. The Plot
ggplot(df_plot_4x1_200, 
       aes(x = condition_label, y = Value_Pct, color = group_type)) +
  
  ## Reference lines (mapped so they appear in legend)
  geom_hline(
    data = ref_lines_4x1_200,
    aes(
      yintercept = intercept,
      linetype   = "Reference\n(Kern et al., 2023)"
    ),
    color = "gray50",
    linewidth = 0.7
  ) +
  
  ## Dots
  geom_point(
    position = position_dodge(width = 0.6),
    size = 3.5
  ) +
  
  ## 4x1 Layout
  facet_grid(facet_group ~ ., scales = "fixed") +
  
  ## Condition colors (FIRST legend)
  scale_color_manual(
    name = "Condition",
    values = c(
      "base"        = "#443983",
      "conf"        = "#31688e",
      "batch"       = "#35b779",
      "batch_conf"  = "#90d743"
    ),
    labels = c(
      "base"       = "Base",
      "conf"       = "+ Confidence",
      "batch"      = "Batch",
      "batch_conf" = "Batch +\nConfidence"
    )
  ) +
  
  ## Reference line legend (SECOND legend)
  scale_linetype_manual(
    name   = NULL,
    values = c("Reference\n(Kern et al., 2023)" = "dashed")
  ) +
  
  ## Y axis
  scale_y_continuous(
    limits = c(0, 100),
    breaks = seq(0, 100, 25),
    labels = function(x) paste0(x, "%")
  ) +
  
  labs(
    x = "Task Structure",
    y = "Prevalence"
  ) +
  
  ## Legend ordering
  guides(
    color    = guide_legend(order = 1, nrow = 2),
    linetype = guide_legend(order = 2)
  ) +
  
  theme_bw(base_size = 12) +
  theme(
    strip.background   = element_blank(),
    strip.text         = element_text(size = 10, face = "bold"),
    legend.position    = "bottom",
    legend.box         = "vertical",
    panel.grid.minor   = element_blank(),
    panel.grid.major.x = element_blank(),
    axis.text.x        = element_text(size = 9)
  )



# save
ggsave("Plots/prevalences_zerofew_ABC_4x1_200subset_dots.pdf",
       last_plot(),
       device = cairo_pdf,
       width  = 12,
       height = 29,
       units  = "cm"
)



################################################################################
### Calculate Rater Agreements 
################################################################################

# get modal ratings
df_agg <- df_long %>%
  group_by(tweet_id, condition, shot) %>%
  summarise(
    modal_HS = Mode(HS),
    modal_OL = Mode(OL),
    .groups = "drop"
  )

# Prepare lists of conditions
conditions    <- sort(unique(df_long$condition))
HS_conditions <- conditions[!grepl("C\\.OL", conditions)]
OL_conditions <- conditions[!grepl("C\\.HS", conditions)]

# Setup empty agreement matrices
across_HS_zero <- matrix(
  NaN,
  nrow = length(HS_conditions),
  ncol = length(HS_conditions),
  dimnames = list(HS_conditions, HS_conditions)
)
across_HS_few <- across_HS_zero
across_OL_zero <- matrix(
  NaN,
  nrow = length(OL_conditions),
  ncol = length(OL_conditions),
  dimnames = list(OL_conditions, OL_conditions)
)
across_OL_few <- across_OL_zero

# Fill in diagonals (Fleiss' kappa) and off-diagonals (Cohen's kappa)
#  Diagonals: Average Within Tweet Agreement within Conditions 
#             --> 3 Ratings per Tweet --> Fleiss Kappa
#  Off-Diagonals: Average agreement of Modal Ratings per Tweet across
#             --> 1 Modal Rating per Condition per Tweet --> Cohens Kappa
for (shot_type in c("zero-shot", "few-shot")) {
  for (cond_i in conditions) {
    # Fleiss' kappa for HS and OL in the same condition
    df_sub <- df_long %>% filter(shot == shot_type, condition == cond_i)
    # get the needed matrix for HS ratings for irr::kappam.fleiss
    # OL
    mat_OL <- df_sub %>%
      select(tweet_id, responder, OL) %>%
      pivot_wider(names_from = responder, values_from = OL) %>%
      select(-tweet_id) %>%
      as.matrix()
    # same for HS
    mat_HS <- df_sub %>%
      select(tweet_id, responder, HS) %>%
      pivot_wider(names_from = responder, values_from = HS) %>%
      select(-tweet_id) %>%
      as.matrix()
    # calculate kappa
    kappa_fleiss_HS <- irr::kappam.fleiss(mat_HS)$value
    kappa_fleiss_OL <- irr::kappam.fleiss(mat_OL)$value
    
    # assign kappa value to the appropriate matrix
    # zero shot
    if (shot_type == "zero-shot") {
      if (cond_i %in% colnames(across_HS_zero)){
        across_HS_zero[cond_i, cond_i] <- kappa_fleiss_HS
      }
      if (cond_i %in% colnames(across_OL_zero)){
        across_OL_zero[cond_i, cond_i] <- kappa_fleiss_OL
      }
    } else {
      # few shot
      if (cond_i %in% colnames(across_HS_zero)){
        across_HS_few[cond_i, cond_i] <- kappa_fleiss_HS
      }
      if (cond_i %in% colnames(across_OL_zero)){
        across_OL_few[cond_i, cond_i] <- kappa_fleiss_OL
      }
    }
    
    # Cohen's kappa between different conditions' modal ratings
    for (cond_j in conditions) {
      if (cond_i != cond_j) {
        # OL between-conditions
        if (cond_i %in% OL_conditions && cond_j %in% OL_conditions) {
          wide_OL <- df_agg %>%
            filter(shot == shot_type, condition %in% OL_conditions) %>%
            select(tweet_id, condition, modal_OL) %>%
            pivot_wider(names_from = condition, values_from = modal_OL)
          pair_ol <- wide_OL %>%
            select(all_of(c(cond_i, cond_j))) %>%
            drop_na()
          kappa_pair_ol <-kappa2(pair_ol)$value 
          if (shot_type=="zero-shot") {
            across_OL_zero[cond_i, cond_j] <- kappa_pair_ol
          } else {
            across_OL_few[cond_i, cond_j]  <- kappa_pair_ol
          }
        }
        # HS between-conditions
        if (cond_i %in% HS_conditions && cond_j %in% HS_conditions) {
          # --> get the modal ratings per condition & tweet
          wide_HS <- df_agg %>%
            filter(shot == shot_type, condition %in% HS_conditions) %>%
            select(tweet_id, condition, modal_HS) %>%
            pivot_wider(names_from = condition, values_from = modal_HS)
          # --> select relevant cols
          pair <- wide_HS %>%
            select(all_of(c(cond_i, cond_j))) %>%
            drop_na()
          # calulcate cohens kappa
          kappa_pair <- kappa2(pair)$value 
          if (shot_type=="zero-shot") {
            across_HS_zero[cond_i, cond_j] <- kappa_pair
          } else {
            across_HS_few[cond_i, cond_j]  <- kappa_pair
          }
        }
      }
    }
  }
}

# convert to dfs
across_HS_zero_df <- as.data.frame(across_HS_zero)
across_HS_few_df  <- as.data.frame(across_HS_few)
across_OL_zero_df <- as.data.frame(across_OL_zero)
across_OL_few_df  <- as.data.frame(across_OL_few)

### get Agreement rates with the Kern et al Reference Data set
# --> calculcate modes for reference data set
kern_modes <- tweets_full %>%
  group_by(tweet_id) %>%
  reframe(
    global_HS = Mode(hate_speech)[1], # select first if multiple modes
    # possible if tweet has only 14 ratings
    global_OL = Mode(offensive_language)[1], 
    .groups = "drop"
  )

df_agg2 <- df_agg %>%
  left_join(kern_modes, by = "tweet_id")

outcomes <- c("HS","OL")
shots    <- unique(df_agg2$shot)
conds    <- unique(df_agg2$condition)

kappa_results <- list()
for (outcome in outcomes) {
  gcol <- paste0("global_", outcome)
  mcol <- paste0("modal_",  outcome)
  kappa_results[[outcome]] <- map_dfr(shots, function(st) {
    tibble(
      condition = conds,
      kappa     = map_dbl(conds, function(cond) {
        tmp <- df_agg2 %>%
          filter(shot == st, condition == cond) %>%
          select(all_of(c(mcol, gcol))) %>%
          drop_na()
        if (nrow(tmp)>0) kappa2(tmp)$value else NA_real_
      }),
      shot = st
    )
  })
}
kappa_results

################################################################################
### Plot Agreements HeatMaps
################################################################################

coolwarm <- colorRampPalette(rev(RColorBrewer::brewer.pal(11, "RdBu")))

# prefixes / suffixes used for the ordering of conditions
prefixes  <- c("A", "B", "C")
suffixes  <- c("", "_conf", "_batch", "_batch_conf")
base_cond_order <- paste0(rep(prefixes, each = length(suffixes)), suffixes)

outcome = 'OL'
shot_code = 'few'

# build one panel (one outcome and shot) and return long data-frame
build_panel_df <- function(outcome = c("OL", "HS"),
                           shot_code = c("zero", "few")) {
  
  outcome   <- match.arg(outcome)
  shot_code <- match.arg(shot_code)     
  
  # select correct matrix
  mat <- get(paste0("across_", outcome, "_", shot_code))
  
  # frop the 'wrong' duplciated C.OL and C.HS cols
  other_outcome <- ifelse(outcome == "HS", "OL", "HS")
  no_cond       <- paste0("C.", other_outcome)
  keep_rows     <- !str_detect(rownames(mat), fixed(no_cond))
  keep_cols     <- !str_detect(colnames(mat), fixed(no_cond))
  mat           <- mat[keep_rows, keep_cols, drop = FALSE]
  
  # remove .OL / .HS 
  strip_pattern <- "\\.(?:OL|HS)"
  rownames(mat) <- str_remove_all(rownames(mat), strip_pattern)
  colnames(mat) <- str_remove_all(colnames(mat), strip_pattern)
  
  # order
  cond_order <- base_cond_order[base_cond_order %in% rownames(mat)]
  mat        <- mat[cond_order, cond_order, drop = FALSE]
  mat_df <- as.data.frame(mat)
  
  # add gap col between agreements and kern agreements
  mat_df$gap <- NaN
  
  kappa_results_select <- kappa_results[[outcome]] 
  kappa_results_select <- kappa_results_select %>% 
    filter(!grepl(other_outcome, condition))
  kappa_results_select$condition <-
    str_remove_all(kappa_results_select$condition, strip_pattern)
  kappa_results_select <- kappa_results_select %>%
    filter(shot == paste0(shot_code, "-shot"),
           condition %in% cond_order) %>%
    select(condition, kappa) %>%
    deframe() 
  kappa_select_df <- as.data.frame(kappa_results_select)
  colnames(kappa_select_df) <- 'reference_agreement'
  
  mat_df <- merge(mat_df,
                  kappa_select_df,
                  by.x = "row.names", by.y = "row.names",
                  sort = FALSE) # keep the correct order
  rownames(mat_df) <- mat_df$Row.names    
  mat_df$Row.names <- NULL 
  
  # long format and only upper triangle
  df <- as_tibble(mat_df, rownames = "row_cond") |>
    pivot_longer(-row_cond,
                 names_to  = "col_cond",
                 values_to = "value") |>
    mutate(outcome = factor(outcome, levels = c("OL", "HS")),
           shot    = factor(paste0(shot_code, "-shot"),
                            levels = c("zero-shot", "few-shot")))
  
  cond_index <- setNames(seq_along(cond_order), cond_order)
  df <- df |>
    mutate(row_idx = cond_index[row_cond],
           col_idx = cond_index[col_cond],
           value   = ifelse(!is.na(row_idx) & !is.na(col_idx) &
                              col_idx > row_idx, NaN, value)) |>
    select(-row_idx, -col_idx)
  
  return(df)
}

#### build the plot data frames
heat_df <- map_dfr(          # loop over outcomes
  c("OL", "HS"),
  function(o) {
    map_dfr(                 # iloop over shot types
      c("zero", "few"),
      function(s) {
        build_panel_df(outcome = o, shot_code = s)
      }
    )
  }
)

all_rows <- unique(heat_df$row_cond)
all_cols <- unique(heat_df$col_cond)

heat_df <- heat_df |>
  mutate(row_cond = factor(row_cond, levels = all_rows),
         col_cond = factor(col_cond, levels = all_cols))


# change the condition names
heat_df <- heat_df %>% 
  mutate(
    outcome  = factor(outcome, levels = c("OL", "HS")), # cols
    shot     = factor(shot,    levels = c("zero-shot",
                                          "few-shot")), # rows
    condition = factor(
      paste(outcome, shot, sep = " | "),
      levels = c("OL | zero-shot", "HS | zero-shot",
                 "OL | few-shot",  "HS | few-shot")     # order
    )
  )


# 1. Update the labeling logic in the dataframe
heat_df_final <- heat_df %>%
  mutate(
    # Create the Task Level
    task_label = case_when(
      grepl("^A", row_cond) ~ "Joint, OL first",
      grepl("^B", row_cond) ~ "Joint, HS first",
      grepl("^C", row_cond) ~ "Separate",
      TRUE ~ ""
    ),
    # Create the Mode Level
    mode_label = case_when(
      grepl("batch_conf", row_cond) ~ "batch + conf",
      grepl("batch", row_cond)      ~ "batch",
      grepl("conf", row_cond)       ~ "conf",
      TRUE                          ~ "base"
    ),
    # Combine them for the Y-axis: "Joint... \n base"
    # The Task label is only shown once (or we just align it)
    clean_row_label = paste0(mode_label),
    clean_col_label = mode_label
  )

# 1. First, ensure the underlying data has the task/mode info accessible
heat_df_clean <- heat_df %>%
  mutate(
    # Extract Task (A, B, C)
    task_prefix = str_extract(row_cond, "^[ABC]"),
    # Rename Task for the display
    task_named = case_when(
      task_prefix == "A" ~ "Joint, OL first",
      task_prefix == "B" ~ "Joint, HS first",
      task_prefix == "C" ~ "Separate",
      TRUE ~ ""
    ),
    # Extract Mode
    mode_part = case_when(
      grepl("batch_conf", row_cond) ~ "batch+conf",
      grepl("batch", row_cond)      ~ "batch",
      grepl("conf", row_cond)       ~ "conf",
      TRUE                          ~ "base"
    ),
    # Create the nested string for the Y-axis: "Joint, OL first | base"
    # This allows us to have one unique string per row while still being able to format it
    nested_row_label = paste(task_named, mode_part, sep = " | ")
  )

# 2. Update the Plot
ggplot(heat_df_clean,
       aes(x = col_cond, y = row_cond, fill = value)) +
  geom_tile(colour = "white", linewidth = 0.4) +              
  geom_text(aes(label = ifelse(is.na(value), "", 
                               sub("^(-?)0\\.", "\\1.", sprintf("%.2f", value))
  )),
  size = 3) +                                      
  
  # Keep your 2x2 layout
  facet_wrap(~ condition, ncol = 2, strip.position = "top") +
  
  # Clean the Y-Axis Labels
  scale_y_discrete(limits = rev, labels = function(x) {
    # Extract Task and Mode from the row string (e.g., "A_conf")
    tasks <- case_when(
      grepl("^A", x) ~ "Joint, OL first",
      grepl("^B", x) ~ "Joint, HS first",
      grepl("^C", x) ~ "Separate",
      TRUE ~ ""
    )
    modes <- case_when(
      grepl("batch_conf", x) ~ "batch+conf",
      grepl("batch", x)      ~ "batch",
      grepl("conf", x)       ~ "conf",
      TRUE                   ~ "base"
    )
    # Combine them for a nested look: "Joint, OL first\nbase"
    paste0(tasks, " - ", modes)
  }) +
  
  # Clean the X-Axis Labels
  scale_x_discrete(labels = function(x) {
    case_when(
      x == "gap" ~ "",
      x == "reference_agreement" ~ "Reference",
      grepl("batch_conf", x) ~ "batch+conf",
      grepl("batch", x)      ~ "batch",
      grepl("conf", x)       ~ "conf",
      TRUE                   ~ "base"
    )
  }) +
  
  scale_fill_gradientn(
    colours  = coolwarm(100),
    limits   = c(0, 1),
    na.value = "white",
    name     = "Kappa"
  ) +
  
  theme_minimal(base_size = 11) +                              
  theme(
    panel.grid      = element_blank(),
    # Rotate X labels 90 degrees
    axis.text.x     = element_text(size = 9, angle = 90, hjust = 1, vjust = 0.5),
    # Keep Y labels horizontal
    axis.text.y     = element_text(size = 8, hjust = 1),
    axis.title      = element_blank(),
    strip.text      = element_text(size = 11, face = "bold"),
    legend.position = "right"
  )

# 1. Update the Plot
ggplot(heat_df,
       aes(x = col_cond, y = row_cond, fill = value)) +
  geom_tile(colour = "white", linewidth = 0.4) +              
  geom_text(aes(label = ifelse(is.na(value), "", 
                               sub("^(-?)0\\.", "\\1.", sprintf("%.2f", value))
  )),
  size = 3.5) +                                      
  
  # 2x2 Layout
  facet_wrap(~ condition, ncol = 2, strip.position = "top") +
  
  # NESTED Y-AXIS (Task label on the 4th row of the block)
  scale_y_discrete(limits = rev, labels = function(y) {
    task_map <- c("A" = "Joint, OL first", "B" = "Joint, HS first", "C" = "Separate")
    task_p   <- str_extract(y, "^[ABC]")
    
    # Counter for vertical blocks
    occ_y <- ave(seq_along(y), task_p, FUN = seq_along)
    
    mode_p <- case_when(
      grepl("batch_conf", y) ~ "batch+conf",
      grepl("batch", y)      ~ "batch",
      grepl("conf", y)       ~ "conf",
      TRUE                   ~ "base"
    )
    
    task_disp <- ifelse(occ_y == 4 & !is.na(task_p), task_map[task_p], "")
    paste0(sprintf("%-18s", task_disp), mode_p)
  }) +
  
  # NESTED X-AXIS (Task label on the 4th column of the block)
  scale_x_discrete(labels = function(x) {
    task_map <- c("A" = "Joint, OL first", "B" = "Joint, HS first", "C" = "Separate")
    task_p   <- str_extract(x, "^[ABC]")
    
    # Counter for horizontal blocks
    occ_x <- ave(seq_along(x), task_p, FUN = seq_along)
    
    mode_p <- case_when(
      x == "gap" ~ "",
      x == "reference_agreement" ~ "Reference",
      grepl("batch_conf", x) ~ "batch+conf",
      grepl("batch", x)      ~ "batch",
      grepl("conf", x)       ~ "conf",
      TRUE                   ~ "base"
    )
    
    # Logic: Show task label ONLY on the 4th column of the block
    # Added spaces after task_disp to separate it from the mode label vertically
    task_disp <- ifelse(occ_x == 1 & !is.na(task_p), 
                        paste0(task_map[task_p], "  "), "")
    
    paste0(task_disp, mode_p)
  }) +
  
  scale_fill_gradientn(
    colours  = coolwarm(100),
    limits   = c(0, 1),
    na.value = "white",
    name     = "Kappa"
  ) +
  
  theme_minimal(base_size = 11) +                              
  theme(
    panel.grid      = element_blank(),
    # X axis: Monospace for alignment
    axis.text.x     = element_text(size = 9, angle = 90, hjust = 1, vjust = 0.5, family = "mono"),
    # Y axis: Monospace for alignment
    axis.text.y     = element_text(size = 9, family = "mono", hjust = 0),
    axis.title      = element_blank(),
    strip.text      = element_text(size = 11, face = "bold"),
    
    # Spacing between the 4 heatmaps
    panel.spacing   = unit(2.5, "lines"),
    legend.position = "right"
  )


# ggsave(
#   "Plots/agreements_new_conditions_2x2.pdf",
#   plot   = last_plot(),
#   device = cairo_pdf,
#   width  = 30,            # Narrower for 2-column fit
#   height = 30,            # Taller to accommodate 4 stacked heatmaps
#   units  = "cm"
# )



### same as above but only 2x1 (few-shot only)
ggplot(heat_df[heat_df$shot == 'few-shot', ],
       aes(x = col_cond, y = row_cond, fill = value)) +
  geom_tile(colour = "white", linewidth = 0.4) +              
  geom_text(aes(label = ifelse(is.na(value), "", 
                               sub("^(-?)0\\.", "\\1.", sprintf("%.2f", value))
  )),
  size = 3.5) +                                      
  
  # 2x2 Layout
  facet_wrap(~ condition, ncol = 1, strip.position = "top") +
  
  # NESTED Y-AXIS (Task label on the 4th row of the block)
  scale_y_discrete(limits = rev, labels = function(y) {
    task_map <- c("A" = "Joint, OL first", "B" = "Joint, HS first", "C" = "Separate")
    task_p   <- str_extract(y, "^[ABC]")
    
    # Counter for vertical blocks
    occ_y <- ave(seq_along(y), task_p, FUN = seq_along)
    
    mode_p <- case_when(
      grepl("batch_conf", y) ~ "batch+conf",
      grepl("batch", y)      ~ "batch",
      grepl("conf", y)       ~ "conf",
      TRUE                   ~ "base"
    )
    
    task_disp <- ifelse(occ_y == 4 & !is.na(task_p), task_map[task_p], "")
    paste0(sprintf("%-18s", task_disp), mode_p)
  }) +
  
  # NESTED X-AXIS (Task label on the 4th column of the block)
  scale_x_discrete(labels = function(x) {
    task_map <- c("A" = "Joint, OL first", "B" = "Joint, HS first", "C" = "Separate")
    task_p   <- str_extract(x, "^[ABC]")
    
    # Counter for horizontal blocks
    occ_x <- ave(seq_along(x), task_p, FUN = seq_along)
    
    mode_p <- case_when(
      x == "gap" ~ "",
      x == "reference_agreement" ~ "Reference",
      grepl("batch_conf", x) ~ "batch+conf",
      grepl("batch", x)      ~ "batch",
      grepl("conf", x)       ~ "conf",
      TRUE                   ~ "base"
    )
    
    # Logic: Show task label ONLY on the 4th column of the block
    # Added spaces after task_disp to separate it from the mode label vertically
    task_disp <- ifelse(occ_x == 1 & !is.na(task_p), 
                        paste0(task_map[task_p], "  "), "")
    
    paste0(task_disp, mode_p)
  }) +
  
  scale_fill_gradientn(
    colours  = coolwarm(100),
    limits   = c(0, 1),
    na.value = "white",
    name     = "Kappa"
  ) +
  
  # Legend at bottom for space efficiency in 4x1 layout
  guides(
    fill = guide_colorbar(
      title = "Kappa (Cohen's or Fleiss')",
      title.position = "top", 
      title.hjust = 0.5,
      barwidth = unit(8, "cm"),
      barheight = unit(0.4, "cm")
    )
  ) +
  
  theme_minimal(base_size = 11) +                              
  theme(
    panel.grid      = element_blank(),
    # X axis: Monospace for alignment
    axis.text.x     = element_text(size = 9, angle = 90, hjust = 1, vjust = 0.5, family = "mono"),
    # Y axis: Monospace for alignment
    axis.text.y     = element_text(size = 9, family = "mono", hjust = 0),
    axis.title      = element_blank(),
    strip.text      = element_text(size = 11, face = "bold"),
    
    # Spacing between the 4 heatmaps
    panel.spacing   = unit(2.5, "lines"),
    legend.position = "bottom"
  )


ggsave(
  "Plots/agreements_few_new_conditions_2x1.pdf",
  plot   = last_plot(),
  device = cairo_pdf,
  width  = 15,            # Narrower for 2-column fit
  height = 20,            # Taller to accommodate 4 stacked heatmaps
  units  = "cm"
)


################################################################################
### Calculate Rater Agreements 
### FOR THE 200 TWEET ZERO SHOT SUBSET
################################################################################

# get modal ratings
df_agg <- df_long_200 %>%
  group_by(tweet_id, condition, shot) %>%
  summarise(
    modal_HS = Mode(HS),
    modal_OL = Mode(OL),
    .groups = "drop"
  )

# Prepare lists of conditions
conditions    <- sort(unique(df_long_200$condition))
HS_conditions <- conditions[!grepl("C\\.OL", conditions)]
OL_conditions <- conditions[!grepl("C\\.HS", conditions)]

# Setup empty agreement matrices
across_HS_zero <- matrix(
  NaN,
  nrow = length(HS_conditions),
  ncol = length(HS_conditions),
  dimnames = list(HS_conditions, HS_conditions)
)
across_HS_few <- across_HS_zero
across_OL_zero <- matrix(
  NaN,
  nrow = length(OL_conditions),
  ncol = length(OL_conditions),
  dimnames = list(OL_conditions, OL_conditions)
)
across_OL_few <- across_OL_zero

# Fill in diagonals (Fleiss' kappa) and off-diagonals (Cohen's kappa)
#  Diagonals: Average Within Tweet Agreement within Conditions 
#             --> 3 Ratings per Tweet --> Fleiss Kappa
#  Off-Diagonals: Average agreement of Modal Ratings per Tweet across
#             --> 1 Modal Rating per Condition per Tweet --> Cohens Kappa
for (shot_type in c("zero-shot", "few-shot")) {
  for (cond_i in conditions) {
    # Fleiss' kappa for HS and OL in the same condition
    df_sub <- df_long_200 %>% filter(shot == shot_type, condition == cond_i)
    # get the needed matrix for HS ratings for irr::kappam.fleiss
    # OL
    mat_OL <- df_sub %>%
      select(tweet_id, responder, OL) %>%
      pivot_wider(names_from = responder, values_from = OL) %>%
      select(-tweet_id) %>%
      as.matrix()
    # same for HS
    mat_HS <- df_sub %>%
      select(tweet_id, responder, HS) %>%
      pivot_wider(names_from = responder, values_from = HS) %>%
      select(-tweet_id) %>%
      as.matrix()
    # calculate kappa
    kappa_fleiss_HS <- irr::kappam.fleiss(mat_HS)$value
    kappa_fleiss_OL <- irr::kappam.fleiss(mat_OL)$value
    
    # assign kappa value to the appropriate matrix
    # zero shot
    if (shot_type == "zero-shot") {
      if (cond_i %in% colnames(across_HS_zero)){
        across_HS_zero[cond_i, cond_i] <- kappa_fleiss_HS
      }
      if (cond_i %in% colnames(across_OL_zero)){
        across_OL_zero[cond_i, cond_i] <- kappa_fleiss_OL
      }
    } else {
      # few shot
      if (cond_i %in% colnames(across_HS_zero)){
        across_HS_few[cond_i, cond_i] <- kappa_fleiss_HS
      }
      if (cond_i %in% colnames(across_OL_zero)){
        across_OL_few[cond_i, cond_i] <- kappa_fleiss_OL
      }
    }
    
    # Cohen's kappa between different conditions' modal ratings
    for (cond_j in conditions) {
      if (cond_i != cond_j) {
        # OL between-conditions
        if (cond_i %in% OL_conditions && cond_j %in% OL_conditions) {
          wide_OL <- df_agg %>%
            filter(shot == shot_type, condition %in% OL_conditions) %>%
            select(tweet_id, condition, modal_OL) %>%
            pivot_wider(names_from = condition, values_from = modal_OL)
          pair_ol <- wide_OL %>%
            select(all_of(c(cond_i, cond_j))) %>%
            drop_na()
          kappa_pair_ol <-kappa2(pair_ol)$value 
          if (shot_type=="zero-shot") {
            across_OL_zero[cond_i, cond_j] <- kappa_pair_ol
          } else {
            across_OL_few[cond_i, cond_j]  <- kappa_pair_ol
          }
        }
        # HS between-conditions
        if (cond_i %in% HS_conditions && cond_j %in% HS_conditions) {
          # --> get the modal ratings per condition & tweet
          wide_HS <- df_agg %>%
            filter(shot == shot_type, condition %in% HS_conditions) %>%
            select(tweet_id, condition, modal_HS) %>%
            pivot_wider(names_from = condition, values_from = modal_HS)
          # --> select relevant cols
          pair <- wide_HS %>%
            select(all_of(c(cond_i, cond_j))) %>%
            drop_na()
          # calulcate cohens kappa
          kappa_pair <- kappa2(pair)$value 
          if (shot_type=="zero-shot") {
            across_HS_zero[cond_i, cond_j] <- kappa_pair
          } else {
            across_HS_few[cond_i, cond_j]  <- kappa_pair
          }
        }
      }
    }
  }
}

# convert to dfs
across_HS_zero_df <- as.data.frame(across_HS_zero)
across_HS_few_df  <- as.data.frame(across_HS_few)
across_OL_zero_df <- as.data.frame(across_OL_zero)
across_OL_few_df  <- as.data.frame(across_OL_few)

### get Agreement rates with the Kern et al Reference Data set
# --> calculcate modes for reference data set
kern_modes <- tweets_full %>%
  group_by(tweet_id) %>%
  reframe(
    global_HS = Mode(hate_speech)[1], # select first if multiple modes
    # possible if tweet has only 14 ratings
    global_OL = Mode(offensive_language)[1], 
    .groups = "drop"
  )

df_agg2 <- df_agg %>%
  left_join(kern_modes, by = "tweet_id")

outcomes <- c("HS","OL")
shots    <- unique(df_agg2$shot)
conds    <- unique(df_agg2$condition)

kappa_results <- list()
for (outcome in outcomes) {
  gcol <- paste0("global_", outcome)
  mcol <- paste0("modal_",  outcome)
  kappa_results[[outcome]] <- map_dfr(shots, function(st) {
    tibble(
      condition = conds,
      kappa     = map_dbl(conds, function(cond) {
        tmp <- df_agg2 %>%
          filter(shot == st, condition == cond) %>%
          select(all_of(c(mcol, gcol))) %>%
          drop_na()
        if (nrow(tmp)>0) kappa2(tmp)$value else NA_real_
      }),
      shot = st
    )
  })
}
kappa_results

################################################################################
### Plot Agreements HeatMaps
################################################################################

coolwarm <- colorRampPalette(rev(RColorBrewer::brewer.pal(11, "RdBu")))

# prefixes / suffixes used for the ordering of conditions
prefixes  <- c("A", "B", "C")
suffixes  <- c("", "_conf", "_batch", "_batch_conf")
base_cond_order <- paste0(rep(prefixes, each = length(suffixes)), suffixes)

outcome = 'OL'
shot_code = 'few'

# build one panel (one outcome and shot) and return long data-frame
build_panel_df <- function(outcome = c("OL", "HS"),
                           shot_code = c("zero", "few")) {
  
  outcome   <- match.arg(outcome)
  shot_code <- match.arg(shot_code)     
  
  # select correct matrix
  mat <- get(paste0("across_", outcome, "_", shot_code))
  
  # frop the 'wrong' duplciated C.OL and C.HS cols
  other_outcome <- ifelse(outcome == "HS", "OL", "HS")
  no_cond       <- paste0("C.", other_outcome)
  keep_rows     <- !str_detect(rownames(mat), fixed(no_cond))
  keep_cols     <- !str_detect(colnames(mat), fixed(no_cond))
  mat           <- mat[keep_rows, keep_cols, drop = FALSE]
  
  # remove .OL / .HS 
  strip_pattern <- "\\.(?:OL|HS)"
  rownames(mat) <- str_remove_all(rownames(mat), strip_pattern)
  colnames(mat) <- str_remove_all(colnames(mat), strip_pattern)
  
  # order
  cond_order <- base_cond_order[base_cond_order %in% rownames(mat)]
  mat        <- mat[cond_order, cond_order, drop = FALSE]
  mat_df <- as.data.frame(mat)
  
  # add gap col between agreements and kern agreements
  mat_df$gap <- NaN
  
  kappa_results_select <- kappa_results[[outcome]] 
  kappa_results_select <- kappa_results_select %>% 
    filter(!grepl(other_outcome, condition))
  kappa_results_select$condition <-
    str_remove_all(kappa_results_select$condition, strip_pattern)
  kappa_results_select <- kappa_results_select %>%
    filter(shot == paste0(shot_code, "-shot"),
           condition %in% cond_order) %>%
    select(condition, kappa) %>%
    deframe() 
  kappa_select_df <- as.data.frame(kappa_results_select)
  colnames(kappa_select_df) <- 'reference_agreement'
  
  mat_df <- merge(mat_df,
                  kappa_select_df,
                  by.x = "row.names", by.y = "row.names",
                  sort = FALSE) # keep the correct order
  rownames(mat_df) <- mat_df$Row.names    
  mat_df$Row.names <- NULL 
  
  # long format and only upper triangle
  df <- as_tibble(mat_df, rownames = "row_cond") |>
    pivot_longer(-row_cond,
                 names_to  = "col_cond",
                 values_to = "value") |>
    mutate(outcome = factor(outcome, levels = c("OL", "HS")),
           shot    = factor(paste0(shot_code, "-shot"),
                            levels = c("zero-shot", "few-shot")))
  
  cond_index <- setNames(seq_along(cond_order), cond_order)
  df <- df |>
    mutate(row_idx = cond_index[row_cond],
           col_idx = cond_index[col_cond],
           value   = ifelse(!is.na(row_idx) & !is.na(col_idx) &
                              col_idx > row_idx, NaN, value)) |>
    select(-row_idx, -col_idx)
  
  return(df)
}

#### build the plot data frames
heat_df <- map_dfr(          # loop over outcomes
  c("OL", "HS"),
  function(o) {
    map_dfr(                 # iloop over shot types
      c("zero", "few"),
      function(s) {
        build_panel_df(outcome = o, shot_code = s)
      }
    )
  }
)

all_rows <- unique(heat_df$row_cond)
all_cols <- unique(heat_df$col_cond)

heat_df <- heat_df |>
  mutate(row_cond = factor(row_cond, levels = all_rows),
         col_cond = factor(col_cond, levels = all_cols))


# change the condition names
heat_df <- heat_df %>% 
  mutate(
    outcome  = factor(outcome, levels = c("OL", "HS")), # cols
    shot     = factor(shot,    levels = c("zero-shot",
                                          "few-shot")), # rows
    condition = factor(
      paste(outcome, shot, sep = " | "),
      levels = c("OL | zero-shot", "HS | zero-shot",
                 "OL | few-shot",  "HS | few-shot")     # order
    )
  )


# 1. Update the labeling logic in the dataframe
heat_df_final <- heat_df %>%
  mutate(
    # Create the Task Level
    task_label = case_when(
      grepl("^A", row_cond) ~ "Joint, OL first",
      grepl("^B", row_cond) ~ "Joint, HS first",
      grepl("^C", row_cond) ~ "Separate",
      TRUE ~ ""
    ),
    # Create the Mode Level
    mode_label = case_when(
      grepl("batch_conf", row_cond) ~ "batch + conf",
      grepl("batch", row_cond)      ~ "batch",
      grepl("conf", row_cond)       ~ "conf",
      TRUE                          ~ "base"
    ),
    # Combine them for the Y-axis: "Joint... \n base"
    # The Task label is only shown once (or we just align it)
    clean_row_label = paste0(mode_label),
    clean_col_label = mode_label
  )

# 1. First, ensure the underlying data has the task/mode info accessible
heat_df_clean <- heat_df %>%
  mutate(
    # Extract Task (A, B, C)
    task_prefix = str_extract(row_cond, "^[ABC]"),
    # Rename Task for the display
    task_named = case_when(
      task_prefix == "A" ~ "Joint, OL first",
      task_prefix == "B" ~ "Joint, HS first",
      task_prefix == "C" ~ "Separate",
      TRUE ~ ""
    ),
    # Extract Mode
    mode_part = case_when(
      grepl("batch_conf", row_cond) ~ "batch+conf",
      grepl("batch", row_cond)      ~ "batch",
      grepl("conf", row_cond)       ~ "conf",
      TRUE                          ~ "base"
    ),
    # Create the nested string for the Y-axis: "Joint, OL first | base"
    # This allows us to have one unique string per row while still being able to format it
    nested_row_label = paste(task_named, mode_part, sep = " | ")
  )


## 4y1 vertical style with new names 
# 1. Force the factor levels for the specific order requested
heat_df <- heat_df %>%
  mutate(condition = factor(condition, levels = c(
    "OL | zero-shot", 
    "HS | zero-shot", 
    "OL | few-shot", 
    "HS | few-shot"
  )))

# 2. Generate the 4x1 Vertical Plot
ggplot(heat_df, aes(x = col_cond, y = row_cond, fill = value)) +
  geom_tile(colour = "white", linewidth = 0.4) +              
  geom_text(aes(label = ifelse(is.na(value), "", 
                               sub("^(-?)0\\.", "\\1.", sprintf("%.2f", value))
  )),
  size = 3.2) + 
  
  # NESTED Y-AXIS (Task label on the 4th row)
  scale_y_discrete(limits = rev, labels = function(y) {
    task_map <- c("A" = "Joint, OL first", "B" = "Joint, HS first", "C" = "Separate")
    task_p   <- str_extract(y, "^[ABC]")
    occ_y    <- ave(seq_along(y), task_p, FUN = seq_along)
    
    mode_p <- case_when(
      grepl("batch_conf", y) ~ "batch+conf",
      grepl("batch", y)      ~ "batch",
      grepl("conf", y)       ~ "conf",
      TRUE                   ~ "base"
    )
    
    task_disp <- ifelse(occ_y == 4 & !is.na(task_p), task_map[task_p], "")
    paste0(sprintf("%-18s", task_disp), mode_p)
  }) +
  
  # NESTED X-AXIS (Task label on the 4th column)
  scale_x_discrete(labels = function(x) {
    task_map <- c("A" = "Joint, OL first", "B" = "Joint, HS first", "C" = "Separate")
    task_p   <- str_extract(x, "^[ABC]")
    occ_x    <- ave(seq_along(x), task_p, FUN = seq_along)
    
    mode_p <- case_when(
      x == "gap" ~ "",
      x == "reference_agreement" ~ "Ref.",
      grepl("batch_conf", x) ~ "batch+conf",
      grepl("batch", x)      ~ "batch",
      grepl("conf", x)       ~ "conf",
      TRUE                   ~ "base"
    )
    
    task_disp <- ifelse(occ_x == 1 & !is.na(task_p), 
                        paste0(task_map[task_p], "  "), "")
    paste0(task_disp, mode_p)
  }) +
  
  scale_fill_gradientn(
    colours  = coolwarm(100),
    limits   = c(0, 1),
    na.value = "white",
    name     = "Kappa"
  ) +
  
  # 4x1 Vertical Layout
  facet_wrap(~ condition, ncol = 1, strip.position = "top") +
  
  # Legend at bottom for space efficiency in 4x1 layout
  guides(
    fill = guide_colorbar(
      title = "Kappa (Cohen's or Fleiss')",
      title.position = "top", 
      title.hjust = 0.5,
      barwidth = unit(8, "cm"),
      barheight = unit(0.4, "cm")
    )
  ) +
  
  theme_minimal(base_size = 11) +                              
  theme(
    panel.grid      = element_blank(),
    axis.text.x     = element_text(size = 9, angle = 90, hjust = 1, vjust = 0.5, family = "mono"),
    axis.text.y     = element_text(size = 9, family = "mono", hjust = 0),
    axis.title      = element_blank(),
    strip.text      = element_text(size = 11, face = "bold"),
    
    legend.position = "bottom",
    panel.spacing   = unit(1.5, "lines"), # Vertical gap between heatmaps
    plot.margin     = margin(10, 10, 10, 10)
  )


ggsave(
  "Plots/agreements_new_conditions_4x1_subset200.pdf",
  plot   = last_plot(),
  device = cairo_pdf,     
  width  = 15,            # Narrower for 2-column fit
  height = 30,            # Taller to accommodate 4 stacked heatmaps
  units  = "cm"
)
