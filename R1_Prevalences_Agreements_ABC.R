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
  path = "Data_Collection/",
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

combined_df_f <- files_f %>%
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

summary_tbl <- df_long %>%
  group_by(condition, shot) %>%
  summarize(
    OL = round(mean(OL, na.rm = TRUE), 2),
    HS = round(mean(HS, na.rm = TRUE), 2),
    .groups = "drop"
  ) %>%
  arrange(condition, shot)

summary_tbl



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



### plot proportions in 2x4 panels
ggplot(df_prop, aes(x = condition, y = Value, fill = Measure)) +
  geom_col(position = position_dodge(0.8), width = 0.7) +
  
  ## reference lines — map colour & linetype so they appear in the legend
  geom_hline(aes(yintercept = overall_OL,
                 colour     = "OL reference",
                 linetype   = "OL reference"),
             size = 0.8) +
  geom_hline(aes(yintercept = overall_HS,
                 colour     = "HS reference",
                 linetype   = "HS reference"),
             size = 0.8) +
  
  ## facet layout 
  facet_grid(
    shot ~ group_type,
    scales = "free_x",
    space  = "free_x",
    switch = "y"
  ) +
  
  ## bars
  scale_fill_manual(
    name   = "Measure",
    values = c(OL = "#1f77b4", HS = "#ff7f0e")
  ) +
  
  ## reference line with same colors
  scale_colour_manual(
    name   = "Reference\n(Kern et al. (2023))",
    values = c("OL reference" = "#1f77b4",
               "HS reference" = "#ff7f0e")
  ) +
  scale_linetype_manual(
    name   = "Reference\n(Kern et al. (2023))",
    values = c("OL reference" = "dashed",
               "HS reference" = "dashed")
  ) +
  
  ## measure legend above reference legend
  guides(
    fill      = guide_legend(order = 1),
    colour    = guide_legend(order = 2),
    linetype  = guide_legend(order = 2)
  ) +
  
  ## y axis labels
  scale_y_continuous(
    limits = c(0, 1),
    breaks = seq(0, 1, by = 0.25),
    labels = scales::percent_format(accuracy = 1)
  ) +
  
  labs(
    x   = NULL,
    y   = NULL          
  ) +
  
  ## theme 
  theme_bw(base_size = 11) +                      
  theme(
    axis.text.x        = element_text(size = 10, angle = 90,
                                      vjust = 0.5, hjust = 1),
    axis.text.y        = element_text(size = 10),
    strip.placement    = "outside",
    strip.background   = element_blank(),
    strip.text         = element_text(size = 11, face = "bold"),
    
    legend.position    = "right",
    legend.box         = "vertical",
    legend.text        = element_text(size = 10),
    legend.title       = element_text(size = 10, face = "bold"),
    legend.background  = element_rect(fill = "transparent",
                                      colour = "transparent"),
    
    panel.grid.major.x = element_blank(),
    panel.grid.minor   = element_blank()
  )


# save as pdf for good vector scaling
ggsave(
  "Plots/R/prevalences_zerofew_ABC.pdf",
  plot   = last_plot(),
  device = cairo_pdf,     
  width  = 29,            # roughly DinA4
  height = 17,             
  units  = "cm"
)



################################################################################
### Same plot but only for few shot conditions
################################################################################

# subset the few-shot rows
df_prop_few <- df_prop %>% filter(grepl('few-shot', shot))


### plot with reference legend 
ggplot(df_prop_few, aes(x = condition, y = Value, fill = Measure)) +
  geom_col(position = position_dodge(0.8), width = 0.7) +
  
  ## reference lines (Kern et al. 2023)
  geom_hline(
    aes(yintercept = overall_OL,
        colour     = "OL reference",
        linetype   = "OL reference"),
    size = 0.8
  ) +
  geom_hline(
    aes(yintercept = overall_HS,
        colour     = "HS reference",
        linetype   = "HS reference"),
    size = 0.8
  ) +
  facet_grid(. ~ group_type, scales = "free_x", space = "free_x") +
  ## actual bars
  scale_fill_manual(
    name   = "Measure",
    values = c(OL = "#1f77b4", HS = "#ff7f0e")
  ) +
  ## rsame colors for reference
  scale_colour_manual(
    name   = "Reference\n(Kern et al. (2023))",
    values = c("OL reference" = "#1f77b4",
               "HS reference" = "#ff7f0e")
  ) +
  scale_linetype_manual(
    name   = "Reference\n(Kern et al. (2023))",
    values = c("OL reference" = "dashed",
               "HS reference" = "dashed")
  ) +
  guides(
    fill      = guide_legend(order = 1),   
    colour    = guide_legend(order = 2),   
    linetype  = guide_legend(order = 2)   
  ) +
  scale_y_continuous(
    limits = c(0, 1),
    breaks = seq(0, 1, by = 0.25),
    labels = scales::percent_format(accuracy = 1)
  ) +
  labs(
    x = NULL,
    y = "Proportion\n(few-shot)"
  ) +
  theme_bw(base_size = 14) +
  theme(
    axis.text.x          = element_text(angle = 45, vjust = 1, hjust = 1),
    strip.placement      = "outside",
    strip.background     = element_blank(),
    legend.position = 'right',
    legend.box = 'vertical', 
    legend.text = element_text(size = 13),
    legend.title = element_text(size = 13, face = 'bold'),
    legend.background    = element_rect(color = "transparent", fill = "transparent"),
    panel.grid.major.x   = element_blank(),
    panel.grid.minor     = element_blank()
  )


# save
ggsave("Plots/R/prevalences_fewshot_ABC.pdf", 
       last_plot(),
       device = cairo_pdf,     
       width  = 29,           
       height = 17,             
       units  = "cm"
)






################################################################################
### Calculate Rater Agreements 
################################################################################

# get modal ratigs
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

### plot
ggplot(heat_df,
       aes(x = col_cond, y = row_cond, fill = value)) +
  geom_tile(colour = "white", linewidth = 0.4) +              
  geom_text(aes(label = ifelse(is.na(value), "", 
                               sub("^(-?)0\\.", "\\1.", sprintf("%.2f", value))
                               )),
            size = 3.5) +                                      
  scale_y_discrete(limits = rev) +
  scale_x_discrete(
    labels = function(x) ifelse(x == "gap", "",  # blank gap
                                ifelse(x == "reference_agreement",
                                       "Reference", x))
  ) +
  scale_fill_gradientn(
    colours  = coolwarm(100),
    limits   = c(0, 1),
    na.value = "white",
    name     = "Kappa\n(Cohen's\nor\nFleiss')"
  ) +
  facet_wrap(~ condition, ncol = 2, strip.position = "top") +
  theme_minimal(base_size = 11) +                             
  theme(
    panel.grid      = element_blank(),
    axis.text.x     = element_text(size = 11, angle = 90, hjust = 1),
    axis.text.y     = element_text(size = 11),
    axis.title      = element_blank(),
    strip.text      = element_text(size = 11, face = "bold"),
    legend.title    = element_text(size = 11, face = "bold"),
    legend.text     = element_text(size = 11),
    plot.title      = element_text(size = 11, face = "bold", hjust = .5),
  ) 


# save as pdf for good vector scaling
ggsave(
  "Plots/R/agreements.pdf",
  plot   = last_plot(),
  device = cairo_pdf,     
  width  = 29,            # close to dina4
  height = 17,             
  units  = "cm"
)


################################################################################
### Inspect some of the low Agreement Cases
################################################################################

## LLM labeks
llm_labels_reshaped <- df_long %>% 
  filter(shot == "few-shot") %>% 
  group_by(tweet_id, condition, .drop = FALSE) %>% 
  summarise(
    mean_OL_llm = mean(OL, na.rm = TRUE),
    mean_HS_llm = mean(HS, na.rm = TRUE),
    .groups     = "drop"
  )

## Human labels
tweets_kern_reshaped <- tweets_full %>% 
  group_by(tweet_id, .drop = FALSE) %>% 
  summarise(
    mean_OL_human = mean(offensive_language, na.rm = TRUE),
    mean_HS_human = mean(hate_speech,        na.rm = TRUE),
    .groups       = "drop"
  )

## actual tweets
tweets_text <- tweets_full %>% 
  distinct(tweet_id, tweet_hashed)

## combine dfs
merged <- llm_labels_reshaped %>% 
  inner_join(tweets_kern_reshaped, by = "tweet_id") %>%  
  left_join(tweets_text,         by = "tweet_id")         

## get the differences between labels 
merged <- merged %>% 
  mutate(
    mean_diff_OL = abs(mean_OL_human - mean_OL_llm),
    mean_diff_HS = abs(mean_HS_human - mean_HS_llm)
  )

set.seed(15)  # random subset of tweets
sample_10 <- merged %>% 
  filter(
    mean_diff_OL > 0.5,
    !str_detect(condition, "^C\\.HS")     
  ) %>% 
  slice_sample(n = 10)

sample_10


################################################################################
### Regression to test the impact of the different conditions
################################################################################

colnames(df_long)

df_long$batch <- ifelse(grepl('batch', df_long$condition), 1, 0)
df_long$conf <- ifelse(grepl('conf', df_long$condition), 1, 0)
df_long <- df_long %>% 
  mutate(
    task = case_when(
      grepl("C\\.OL", condition) ~ "C.OL",
      grepl("C\\.HS", condition) ~ "C.HS",
      grepl("A",     condition)  ~ "A",
      grepl("B",     condition)  ~ "B",
      TRUE                       ~ NA_character_     # fallback
    )
  )

# GLMM for OL

colnames(df_long)
model_OL <- glmer(OL ~ 1 + task + batch + conf + (1 | tweet_id),
                  data = df_long[!grepl('C.HS', df_long$condition) , ],
                  family = binomial
                  , control = glmerControl(optimizer = "bobyqa")
                  )
summary(model_OL)
# lower prob for OL in task B and C 
#     --> in B: OL is asked second
# lower prob for B in batch
# higher prob when asking for confidence 

# GLMM for HS
model_HS <- glmer(HS ~ 1 + task + batch + conf + (1 | tweet_id),
                  data = df_long[!grepl('C.OL', df_long$condition) , ],
                  family = binomial
                  , control = glmerControl(optimizer = "bobyqa")
)

summary(model_HS)
# higher prob for HS in task B and C
#     --> in B: HS is asked first
# lower prob for B in batch
# higher prob when asking for confidence 









