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


# Prepare Data
df_filtered <- df_long[!grepl('C.HS', df_long$condition), ]
df_filtered$task <- droplevels(as.factor(df_filtered$task))
contrasts(df_filtered$task) <- contr.sum(levels(df_filtered$task))

df_filtered$task  <- as.factor(df_filtered$task)
df_filtered$batch <- as.factor(df_filtered$batch)
df_filtered$conf  <- as.factor(df_filtered$conf)


# Linear Probability Model
library(lme4)
library(broom.mixed)
colnames(df_long)
library(lmerTest)
library(sjPlot)
model_OL_linprob <- lmer(OL ~ 0 + task + batch + conf + (1 | tweet_id),
                         data = df_filtered,
)
# Table for Main Paper
tab_model(model_OL_linprob)



#### GLMMs with Marginalization over Random Effects
# GLMM for OL with reference categories
# reference is task C (separate labeling)
df_filtered$task <- relevel(df_filtered$task, ref = "C.OL")
model_OL_refC <- glmer(OL ~ 1 + task + batch + conf + (1 | tweet_id),
                       data = df_filtered,
                       family = binomial(link = "logit")
                       , control = glmerControl(optimizer = "bobyqa")
)
summary(model_OL_refC)


# glmm w/o intercept 
model_OL <- glmer(OL ~ 0 + task + batch + conf + (1 | tweet_id),
                  data = df_filtered,
                  family = binomial(link = "logit")
                  , control = glmerControl(optimizer = "bobyqa")
                  )
summary(model_OL)


# get task differences
library(emmeans)
task_means <- emmeans(model_OL_linprob, "task")
print(task_means)

# Compare all tasks against each other
task_diffs <- pairs(task_means, adjust = "tukey")
print(task_diffs)


# 1. Get the log-odds (estimates) for each task
library(emmeans)
task_means <- emmeans(model_OL, ~ task)
print(task_means)
conf_means <- emmeans(model_OL, ~ conf)
print(conf_means)
batch_means <- emmeans(model_OL, ~ batch)
print(batch_means)

overall_means <- emmeans(model_OL, ~ task + batch + conf)
print(overall_means)

grand_mean <- emmeans(model_OL, ~ 1)
grand_mean
cell_means <- emmeans(model_OL, ~ task + batch + conf)

diff_from_global <- contrast(cell_means, method = "eff")
diff_from_global

task_coef <- contrast(
  emmeans(model_OL, ~ task),
  method = "eff"
)
task_coef

# 2. To see the "deviations" (how much each task differs from the global intercept)
# This subtracts the grand mean from each task mean
eff_means_task <- contrast(task_means, "eff")
print(eff_means_task)
eff_means_conf <- contrast(conf_means, "eff")
print(eff_means_conf)
eff_means_batch <- contrast(batch_means, "eff")
print(eff_means_batch)


# Standard GLM
model_OL_glm <- glm(OL ~ 0 + task + batch + conf,
                    data = df_filtered,
                    family = binomial(link = "logit"))
summary(model_OL_glm)


tab_model(model_OL_glm,                
          transform = NULL,         
          show.intercept = FALSE)   



#############
# Modeling HS Labels 
#############

# Preparing Data 
df_filtered <- df_long[!grepl('C.OL', df_long$condition), ]
df_filtered$task <- droplevels(as.factor(df_filtered$task))
contrasts(df_filtered$task) <- contr.sum(levels(df_filtered$task))

df_filtered$task  <- as.factor(df_filtered$task)
df_filtered$batch <- as.factor(df_filtered$batch)
df_filtered$conf  <- as.factor(df_filtered$conf)

# linear probability model
model_HS_linprob <- lmer(HS ~ 0 + task + batch + conf + (1 | tweet_id),
                         data = df_filtered,
)
summary(model_HS_linprob)

tab_model(model_HS_linprob) # Table of Main Paper 


#### GLMMs with Marginalization over Random Effects
# GLMM for OL with reference categories
# reference is task C (separate labeling)
model_HS <- glmer(HS ~ 0 + task + batch + conf + (1 | tweet_id),
                  data = df_filtered,
                  family = binomial
                  , control = glmerControl(optimizer = "bobyqa")
)
summary(model_HS)

# get task differences
library(emmeans)
task_means <- emmeans(model_HS_linprob, "task")
print(task_means)

# Compare all tasks against each other
task_diffs <- pairs(task_means, adjust = "tukey")
print(task_diffs)

# 1. Get the log-odds (estimates) for each task
library(emmeans)
task_means <- emmeans(model_HS, ~ task)
print(task_means)
conf_means <- emmeans(model_HS, ~ conf)
print(conf_means)
batch_means <- emmeans(model_HS, ~ batch)
print(batch_means)

overall_means <- emmeans(model_HS, ~ task + batch + conf)
print(overall_means)

grand_mean <- emmeans(model_HS, ~ 1)
grand_mean
cell_means <- emmeans(model_HS, ~ task + batch + conf)

diff_from_global <- contrast(cell_means, method = "eff")
diff_from_global

task_coef <- contrast(
  emmeans(model_HS, ~ task),
  method = "eff"
)
task_coef


# Standard GLM
model_HS_glm <- glm(HS ~ 0 + task + batch + conf,
                    data = df_filtered,
                    family = binomial(link = "logit"))
summary(model_HS_glm)

