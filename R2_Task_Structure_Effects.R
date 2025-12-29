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
### Load and RBind the GPT Labelled Tweets
################################################################################

# few shot (main sample)
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
  select(batch_id, tweet_in_batch, tweet_id, tweet, condition, shot, 
         starts_with("R")) %>%
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
      TRUE                       ~ NA_character_  
    )
  )

# GLMM for OL
library(lme4)
library(broom.mixed)
colnames(df_long)
model_OL <- glmer(OL ~ 1 + task + batch + conf + (1 | tweet_id),
                  data = df_long[!grepl('C.HS', df_long$condition) , ],
                  family = binomial
                  , control = glmerControl(optimizer = "bobyqa")
                  )
summary(model_OL)
options(scipen = 9)
tidy(model_OL,conf.int=TRUE,exponentiate=FALSE,effects="fixed") %>% 
  mutate(across(
    .cols = where(is.numeric),
    .fns  = ~ round(.x, 3))) |> 
  as.data.frame() |> print()
# lower prob for OL in task B and C 
#     --> in B: OL is asked second
# lower prob for B in batch
# higher prob when asking for confidence 

tab_model(model_OL,                 # get the table for the paper 
          transform = NULL,         # betas 
          show.intercept = FALSE)   # no intercept term



# GLMM for HS
model_HS <- glmer(HS ~ 1 + task + batch + conf + (1 | tweet_id),
                  data = df_long[!grepl('C.OL', df_long$condition) , ],
                  family = binomial
                  , control = glmerControl(optimizer = "bobyqa")
)

summary(model_HS)

tidy(model_HS,conf.int=TRUE,exponentiate=FALSE,effects="fixed") %>% 
  mutate(across(
    .cols = where(is.numeric),
    .fns  = ~ round(.x, 3))) |> 
  as.data.frame() |> print()


# higher prob for HS in task B and C
#     --> in B: HS is asked first
# lower prob for B in batch
# higher prob when asking for confidence 


tab_model(model_HS,                 # get the table for the paper 
          transform = NULL,         # betas 
          show.intercept = FALSE)   # no intercept term








################################################################################
###' Repeated Regression with Task Condition C as baseline
################################################################################

colnames(df_long)

df_long$batch <- ifelse(grepl('batch', df_long$condition), 1, 0)
df_long$conf <- ifelse(grepl('conf', df_long$condition), 1, 0)

df_long$batch_f <- factor(ifelse(grepl("batch", df_long$condition), 
                                 "batch", "noBatch"))
df_long$conf_f  <- factor(ifelse(grepl("conf",  df_long$condition), 
                                 "conf",  "noConf"))


df_long <- df_long %>% 
  mutate(
    task = case_when(
      grepl("C\\.OL", condition) ~ "C.OL",
      grepl("C\\.HS", condition) ~ "C.HS",
      grepl("A",     condition)  ~ "A",
      grepl("B",     condition)  ~ "B",
      TRUE                       ~ NA_character_  
    )
  )

# GLMM for OL
library(lme4)
library(broom.mixed)
colnames(df_long)

df_long_OL <- df_long[!grepl('C.HS', df_long$condition) , ]
df_long_OL$task <- factor(df_long_OL$task, levels = c("C.OL", "A", "B"))

model_OL_ref_c <- glmer(OL ~ 1 + task + batch + conf + (1 | tweet_id),
                  data = df_long_OL,
                  family = binomial
                  , control = glmerControl(optimizer = "bobyqa")
)
summary(model_OL_ref_c)
options(scipen = 9)
tidy(model_OL_ref_c,conf.int=TRUE,exponentiate=FALSE,effects="fixed") %>% 
  mutate(across(
    .cols = where(is.numeric),
    .fns  = ~ round(.x, 3))) |> 
  as.data.frame() |> print()
# lower prob for OL in task B and C 
#     --> in B: OL is asked second
# lower prob for B in batch
# higher prob when asking for confidence 

tab_model(model_OL_ref_c,           # get the table for the paper 
          transform = NULL,         # betas 
          show.intercept = FALSE)   # no intercept term



# GLMM for HS
df_long_HS <- df_long[!grepl('C.OL', df_long$condition) , ]
df_long_HS$task <- factor(df_long_HS$task, levels = c("C.HS", "A", "B"))
model_HS_ref_c <- glmer(HS ~ 1 + task + batch + conf + (1 | tweet_id),
                  data = df_long_HS,
                  family = binomial
                  , control = glmerControl(optimizer = "bobyqa")
)

summary(model_HS_ref_c)

tidy(model_HS_ref_c,conf.int=TRUE,exponentiate=FALSE,effects="fixed") %>% 
  mutate(across(
    .cols = where(is.numeric),
    .fns  = ~ round(.x, 3))) |> 
  as.data.frame() |> print()


# higher prob for HS in task B and C
#     --> in B: HS is asked first
# lower prob for B in batch
# higher prob when asking for confidence 


tab_model(model_HS_ref_c,           # get the table for the paper 
          transform = NULL,         # betas 
          show.intercept = FALSE)   # no intercept term






