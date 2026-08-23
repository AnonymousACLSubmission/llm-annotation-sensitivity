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
library(patchwork)   
library(RColorBrewer)
library(data.table)
library(purrr)
library(lme4)
library(readr)
library(tibble)        
library(psych)         


### Set WD and clear environment
setwd(dirname(rstudioapi::getActiveDocumentContext()$path)) # wd = source loc
rm(list = ls())

# source the certainty preprocessing from R3a_Certainties
# ~2 mins
# source('R3a_Certainties.R')

# remove non-needed objects
rm(list = setdiff(ls(envir = .GlobalEnv), 
                  c("confidence_df_final",    # conf_conditions --> 18K rows 
                    "entropy_df_full",        # full data set --> 36K rows
                    "disparity_df_means")),   # non batch --> 18K rows 
   envir = .GlobalEnv)




################################################################################
### Load and RBind the GPT Labelled Tweets as in Task Structure Effects Script
################################################################################

# few shot
date_to_import_f <- "2025_07_01"
files_f <- list.files(
  path = "Data_Collection/OL_NH/",
  pattern = paste0("^", date_to_import_f),
  full.names = TRUE
)

combined_df <- files_f %>%
  map_df(read_csv, .id = "source") %>%
  arrange(batch_id, tweet_in_batch) %>%
  mutate(shot = "few-shot") %>%
  select(1:4, shot, everything())

combined_df$cond_clean <- gsub("\\.(OL|HS)", "", combined_df$condition)


################################################################################
### Use GPT Labels from combined df 
### merge with confidence metrics
### for each outcome and metric combiation, remove the top 10% uncertain tweets
################################################################################

shared_cols <- c(
  "batch_id", "tweet_in_batch", "tweet_id", 
  "tweet", "condition", "shot"
)

##################
#####' merge *confidence* scores to tweets and only keep *top 90%*
#####' higher is better here
combined_df_confidence <- merge(combined_df,
                                confidence_df_final[,c("tweet_id",
                                                       "cond_clean",
                                                       "mean_tweet_conf_OL",
                                                       "mean_tweet_conf_HS")],
                                by = c("tweet_id", "cond_clean"))

# get tweets with highest mean mean_tweet_confidence_OL across conditions
top_conf_ids <- combined_df_confidence %>%
  filter(grepl('A|B|C.OL', condition)) %>%
  group_by(tweet_id) %>%
  summarise(MEAN_mean_tweet_conf_OL = mean(mean_tweet_conf_OL, na.rm = TRUE)) %>%
  slice_max(order_by = MEAN_mean_tweet_conf_OL, prop = 0.9, with_ties = F) %>%
  pull(tweet_id) 
combined_df_confidence_top90_OL <- combined_df_confidence %>%
  filter(grepl('A|B|C.OL', condition)) %>%
  filter(tweet_id %in% top_conf_ids)


top_conf_ids <- combined_df_confidence %>%
  filter(grepl('A|B|C.HS', condition)) %>%
  group_by(tweet_id) %>%
  summarise(MEAN_mean_tweet_conf_HS = mean(mean_tweet_conf_HS, na.rm = TRUE)) %>%
  slice_max(order_by = MEAN_mean_tweet_conf_HS, prop = 0.9, with_ties = F) %>%
  pull(tweet_id)
combined_df_confidence_top90_HS <- combined_df_confidence %>%
  filter(grepl('A|B|C.HS', condition)) %>%
  filter(tweet_id %in% top_conf_ids)



# wide to long
df_long_confidence_top90_OL <- combined_df_confidence_top90_OL %>%
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

df_long_confidence_top90_HS <- combined_df_confidence_top90_HS %>%
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

##################
#####' merge *entropy* scores to tweets and only keep *bottom 90%*
#####' lower is better here
combined_df_entropy <- merge(combined_df,
                                entropy_df_full[,c("tweet_id",
                                                       "cond_clean",
                                                       "tweet_entropy_OL",
                                                       "tweet_entropy_HS")],
                                by = c("tweet_id", "cond_clean"))

# get tweets with lowest mean_tweet_entropy_OL across conditions
low_entropy_ids <- combined_df_entropy %>%
  filter(grepl('A|B|C.OL', condition)) %>%
  group_by(tweet_id) %>%
  summarise(mean_tweet_entropy_OL = mean(tweet_entropy_OL, na.rm = TRUE)) %>%
  slice_min(order_by = mean_tweet_entropy_OL, prop = 0.1, with_ties = F) %>%
  pull(tweet_id) 
combined_df_entropy_top90_OL <- combined_df_entropy %>%
  filter(grepl('A|B|C.OL', condition)) %>%
  filter(!tweet_id %in% low_entropy_ids)



low_entropy_ids <- combined_df_entropy %>%
  filter(grepl('A|B|C.HS', condition)) %>%
  group_by(tweet_id) %>%
  summarise(mean_tweet_entropy_HS = mean(tweet_entropy_HS, na.rm = TRUE)) %>%
  slice_min(order_by = mean_tweet_entropy_HS, prop = 0.1, with_ties = F) %>%
  pull(tweet_id) 
combined_df_entropy_top90_HS <- combined_df_entropy %>%
  filter(grepl('A|B|C.HS', condition)) %>%
  filter(!tweet_id %in% low_entropy_ids)


# wide to long
df_long_entropy_top90_OL <- combined_df_entropy_top90_OL %>%
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

df_long_entropy_top90_HS <- combined_df_entropy_top90_HS %>%
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


##################
#####' merge *disparity* scores to tweets and only keep *top 90%*
#####' higher is better here
combined_df_disparity <- merge(combined_df,
                               disparity_df_means[,c("tweet_id",
                                                   "cond_clean",
                                                   "mean_disparity_OL",
                                                   "mean_disparity_HS")],
                                by = c("tweet_id", "cond_clean"))

# remove duplicated condition rows (only the C.OLs stay)
# dupes <- duplicated(combined_df_disparity[,c('tweet_id', 'cond_clean')])
# combined_df_disparity <- combined_df_disparity[!dupes,]

# get tweets with highest mean mean_tweet_confidence_OL across conditions
top_disp_ids <- combined_df_disparity %>%
  filter(grepl('A|B|C.OL', condition)) %>%
  group_by(tweet_id) %>%
  summarise(MEAN_mean_disparity_OL = mean(mean_disparity_OL, na.rm = TRUE)) %>%
  slice_max(order_by = MEAN_mean_disparity_OL, prop = 0.9, with_ties = F) %>%
  pull(tweet_id) 
combined_df_disparity_top90_OL <- combined_df_disparity %>%
  filter(grepl('A|B|C.OL', condition)) %>%
  filter(tweet_id %in% top_disp_ids)

top_disp_ids <- combined_df_disparity %>%
  filter(grepl('A|B|C.HS', condition)) %>%
  group_by(tweet_id) %>%
  summarise(MEAN_mean_disparity_HS = mean(mean_disparity_HS, na.rm = TRUE)) %>%
  slice_max(order_by = MEAN_mean_disparity_HS, prop = 0.9, with_ties = F) %>%
  pull(tweet_id) 
combined_df_disparity_top90_HS <- combined_df_disparity %>%
  filter(grepl('A|B|C.HS', condition)) %>%
  filter(tweet_id %in% top_disp_ids)


# wide to long
df_long_disparity_top90_OL <- combined_df_disparity_top90_OL %>%
  select(batch_id, tweet_in_batch, tweet_id, tweet, condition, cond_clean ,
         shot, starts_with("R")) %>%
  pivot_longer(
    cols      = starts_with("R"),
    names_to  = c("responder", ".value"),
    names_sep = "_"
  ) %>%
  mutate(
    OL = as.integer(str_detect(label, "(?<!N)OL")),
    HS = as.integer(str_detect(label, "(?<!N)HS"))
  )

df_long_disparity_top90_HS <- combined_df_disparity_top90_HS %>%
  select(batch_id, tweet_in_batch, tweet_id, tweet, condition, cond_clean, 
         shot, starts_with("R")) %>%
  pivot_longer(
    cols      = starts_with("R"),
    names_to  = c("responder", ".value"),
    names_sep = "_"
  ) %>%
  mutate(
    OL = as.integer(str_detect(label, "(?<!N)OL")),
    HS = as.integer(str_detect(label, "(?<!N)HS"))
  )


################################################################################
### Add the necessary factor vars to the long dfs
################################################################################

# Confidence DFs (only in conf_dfs)
colnames(df_long_confidence_top90_OL)

df_long_confidence_top90_OL$batch <- 
  ifelse(grepl('batch', df_long_confidence_top90_OL$condition), 1, 0)
df_long_confidence_top90_HS$batch <- 
  ifelse(grepl('batch', df_long_confidence_top90_HS$condition), 1, 0)

df_long_confidence_top90_OL$conf <- 
  ifelse(grepl('conf', df_long_confidence_top90_OL$condition), 1, 0)
df_long_confidence_top90_HS$conf <- 
  ifelse(grepl('conf', df_long_confidence_top90_HS$condition), 1, 0)

df_long_confidence_top90_OL <- df_long_confidence_top90_OL %>% 
  mutate(
    task = case_when(
      grepl("C\\.OL", condition) ~ "C.OL",
      grepl("C\\.HS", condition) ~ "C.HS",
      grepl("A",     condition)  ~ "A",
      grepl("B",     condition)  ~ "B",
      TRUE                       ~ NA_character_  
    )
  )

df_long_confidence_top90_HS <- df_long_confidence_top90_HS %>% 
  mutate(
    task = case_when(
      grepl("C\\.OL", condition) ~ "C.OL",
      grepl("C\\.HS", condition) ~ "C.HS",
      grepl("A",     condition)  ~ "A",
      grepl("B",     condition)  ~ "B",
      TRUE                       ~ NA_character_  
    )
  )

# Entropy DFs (all rows)
colnames(df_long_entropy_top90_OL)

df_long_entropy_top90_OL$batch <- 
  ifelse(grepl('batch', df_long_entropy_top90_OL$condition), 1, 0)
df_long_entropy_top90_HS$batch <- 
  ifelse(grepl('batch', df_long_entropy_top90_HS$condition), 1, 0)

df_long_entropy_top90_OL$conf <- 
  ifelse(grepl('conf', df_long_entropy_top90_OL$condition), 1, 0)
df_long_entropy_top90_HS$conf <- 
  ifelse(grepl('conf', df_long_entropy_top90_HS$condition), 1, 0)

df_long_entropy_top90_OL <- df_long_entropy_top90_OL %>% 
  mutate(
    task = case_when(
      grepl("C\\.OL", condition) ~ "C.OL",
      grepl("C\\.HS", condition) ~ "C.HS",
      grepl("A",     condition)  ~ "A",
      grepl("B",     condition)  ~ "B",
      TRUE                       ~ NA_character_  
    )
  )

df_long_entropy_top90_HS <- df_long_entropy_top90_HS %>% 
  mutate(
    task = case_when(
      grepl("C\\.OL", condition) ~ "C.OL",
      grepl("C\\.HS", condition) ~ "C.HS",
      grepl("A",     condition)  ~ "A",
      grepl("B",     condition)  ~ "B",
      TRUE                       ~ NA_character_  
    )
  )

# Disparity DFs (only non-batch rows)
colnames(df_long_disparity_top90_OL)

df_long_disparity_top90_OL$batch <- 
  ifelse(grepl('batch', df_long_disparity_top90_OL$condition), 1, 0)
df_long_disparity_top90_HS$batch <- 
  ifelse(grepl('batch', df_long_disparity_top90_HS$condition), 1, 0)

df_long_disparity_top90_OL$conf <- 
  ifelse(grepl('conf', df_long_disparity_top90_OL$condition), 1, 0)
df_long_disparity_top90_HS$conf <- 
  ifelse(grepl('conf', df_long_disparity_top90_HS$condition), 1, 0)

df_long_disparity_top90_OL <- df_long_disparity_top90_OL %>% 
  mutate(
    task = case_when(
      grepl("C\\.OL", condition) ~ "C.OL",
      grepl("C\\.HS", condition) ~ "C.HS",
      grepl("A",     condition)  ~ "A",
      grepl("B",     condition)  ~ "B",
      TRUE                       ~ NA_character_  
    )
  )

df_long_disparity_top90_HS <- df_long_disparity_top90_HS %>% 
  mutate(
    task = case_when(
      grepl("C\\.OL", condition) ~ "C.OL",
      grepl("C\\.HS", condition) ~ "C.HS",
      grepl("A",     condition)  ~ "A",
      grepl("B",     condition)  ~ "B",
      TRUE                       ~ NA_character_  
    )
  )



################################################################################
### Fit the LPM regression models with the top 90%
################################################################################

#' *Confidence* (conf conditions only)
# GLMM for OL 
library(lme4)
colnames(df_long_confidence_top90_OL)
model_OL_conf_linprob <- lmer(OL ~ 0 + task + batch + (1 | tweet_id),
                       data = df_long_confidence_top90_OL[!grepl('C.HS', 
                                                                 df_long_confidence_top90_OL$condition) , ]
)
summary(model_OL_conf_linprob)

tab_model(model_OL_conf_linprob,            # get the table for the paper appendix
          transform = NULL,         # betas 
          show.intercept = FALSE)   # no intercept term


# GLMM for HS
model_HS_conf_linprob <- lmer(HS ~ 0 + task + batch + (1 | tweet_id),
                       data = df_long_confidence_top90_HS[!grepl('C.OL', 
                                                                 df_long_confidence_top90_HS$condition) , ]
)

summary(model_HS_conf_linprob)

tab_model(model_HS_conf_linprob,            # get the table for the paper appendix
          # transform = NULL,         # betas 
          show.intercept = FALSE)   # no intercept term



#' *Entropy* (all conditions)
# GLMM for OL
colnames(df_long_entropy_top90_OL)
model_OL_entr_linprob <- lmer(OL ~ 0 + task + batch + conf +(1 | tweet_id),
                       data = df_long_entropy_top90_OL[!grepl('C.HS', 
                                                              df_long_entropy_top90_OL$condition) , ]
)
summary(model_OL_entr_linprob)

tab_model(model_OL_entr_linprob,            # get the table for the paper appendix
          transform = NULL,         # betas 
          show.intercept = FALSE)   # no intercept term

# GLMM for HS
model_HS_entr_linprob <- lmer(HS ~ 0 + task + batch + conf + (1 | tweet_id),
                       data = df_long_entropy_top90_HS[!grepl('C.OL', 
                                                              df_long_entropy_top90_HS$condition), ]
)
summary(model_HS_entr_linprob)

tab_model(model_HS_entr_linprob,            # get the table for the paper appendix
          transform = NULL,         # betas 
          show.intercept = FALSE)   # no intercept term



#' *Disparity* (non batch coniditions only)
# GLMM for OL
colnames(df_long_disparity_top90_OL)
model_OL_disp_linprob <- lmer(OL ~ 0 + task + conf + (1 | tweet_id),
                       data = df_long_disparity_top90_OL[!grepl('C.HS', 
                                                                df_long_disparity_top90_OL$condition), ]
)
summary(model_OL_disp_linprob)

tab_model(model_OL_disp_linprob,            # get the table for the paper appendix
          transform = NULL,         # betas 
          show.intercept = FALSE)   # no intercept term

# GLMM for HS
model_HS_disp_linprob <- lmer(HS ~ 0 + task + conf + (1 | tweet_id),
                       data = df_long_disparity_top90_HS[!grepl('C.OL', 
                                                                df_long_disparity_top90_HS$condition), ]
)

summary(model_HS_disp_linprob)

tab_model(model_HS_disp_linprob,            # get the table for the paper appendix
          transform = NULL,         # betas 
          show.intercept = FALSE)   # no intercept term
