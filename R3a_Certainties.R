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


################################################################################
### Load, RBind, and restructure the GPT Labelled Tweets
################################################################################

csv_paths <- list.files("Data_Collection/OL_NH/", 
                        pattern = "\\.csv$", full.names = TRUE)
csv_paths <- csv_paths[grepl('2025_07_01', csv_paths)]
dfs <- lapply(csv_paths, read_csv, show_col_types = FALSE)
all_data <- bind_rows(dfs)
all_data <- as.data.frame(all_data)
length(unique(all_data$tweet_id))
length(unique(all_data$batch_id))
sum(duplicated(all_data))

# add few shot col
all_data$shot = 'few-shot'

## cond_clean col
all_data$cond_clean <- gsub("\\.(OL|HS)", "", all_data$condition)


################################################################################
###' *CERTAINTY METRIC 1: SELF-REPORTED CONFIDENCE SCORES*
### --> confidence conditions only
################################################################################

confidence_df <- all_data %>% filter(grepl('conf', condition))

confidence_df <- confidence_df %>%
  mutate(
    R1_OL = as.integer(str_detect(R1_label, "(?<!N)OL")),
    R2_OL = as.integer(str_detect(R2_label, "(?<!N)OL")),
    R3_OL = as.integer(str_detect(R3_label, "(?<!N)OL")),
    R1_HS = as.integer(str_detect(R1_label, "(?<!N)HS")),
    R2_HS = as.integer(str_detect(R2_label, "(?<!N)HS")),
    R3_HS = as.integer(str_detect(R3_label, "(?<!N)HS")),
  )


confidence_df$conf_R1 <- regmatches(confidence_df$R1_score, 
                              gregexpr("[[:digit:]]+", 
                                       confidence_df$R1_score)) 
confidence_df$conf_R2 <- regmatches(confidence_df$R2_score, 
                              gregexpr("[[:digit:]]+", 
                                       confidence_df$R2_score)) 
confidence_df$conf_R3 <- regmatches(confidence_df$R3_score, 
                              gregexpr("[[:digit:]]+", 
                                       confidence_df$R3_score)) 

# Correctly order the confidences
# takes some time... (not the most efficient approach but straightforward) 
for (row in 1:nrow(confidence_df)){
  if (grepl('A',confidence_df[row, 'condition'])){
    confidence_df[row, 'R1_OL_confidence'] <- unlist(confidence_df[row, 'conf_R1'][[1]])[1]
    confidence_df[row, 'R1_HS_confidence'] <- unlist(confidence_df[row, 'conf_R1'][[1]])[2]
    confidence_df[row, 'R2_OL_confidence'] <- unlist(confidence_df[row, 'conf_R2'][[1]])[1]
    confidence_df[row, 'R2_HS_confidence'] <- unlist(confidence_df[row, 'conf_R2'][[1]])[2]
    confidence_df[row, 'R3_OL_confidence'] <- unlist(confidence_df[row, 'conf_R3'][[1]])[1]
    confidence_df[row, 'R3_HS_confidence'] <- unlist(confidence_df[row, 'conf_R3'][[1]])[2]
  } else if (grepl('B',confidence_df[row, 'condition'])){
    confidence_df[row, 'R1_OL_confidence'] <- unlist(confidence_df[row, 'conf_R1'][[1]])[2]
    confidence_df[row, 'R1_HS_confidence'] <- unlist(confidence_df[row, 'conf_R1'][[1]])[1]
    confidence_df[row, 'R2_OL_confidence'] <- unlist(confidence_df[row, 'conf_R2'][[1]])[2]
    confidence_df[row, 'R2_HS_confidence'] <- unlist(confidence_df[row, 'conf_R2'][[1]])[1]
    confidence_df[row, 'R3_OL_confidence'] <- unlist(confidence_df[row, 'conf_R3'][[1]])[2]
    confidence_df[row, 'R3_HS_confidence'] <- unlist(confidence_df[row, 'conf_R3'][[1]])[1]
  } else if (grepl('C.OL',confidence_df[row, 'condition'])){
    confidence_df[row, 'R1_OL_confidence'] <- unlist(confidence_df[row, 'conf_R1'][[1]])[1]
    confidence_df[row, 'R1_HS_confidence'] <- NA
    confidence_df[row, 'R2_OL_confidence'] <- unlist(confidence_df[row, 'conf_R2'][[1]])[1]
    confidence_df[row, 'R2_HS_confidence'] <- NA
    confidence_df[row, 'R3_OL_confidence'] <- unlist(confidence_df[row, 'conf_R3'][[1]])[1]
    confidence_df[row, 'R3_HS_confidence'] <- NA
  } else if (grepl('C.HS',confidence_df[row, 'condition'])){
    confidence_df[row, 'R1_OL_confidence'] <- NA
    confidence_df[row, 'R1_HS_confidence'] <- unlist(confidence_df[row, 'conf_R1'][[1]])[1]
    confidence_df[row, 'R2_OL_confidence'] <- NA
    confidence_df[row, 'R2_HS_confidence'] <- unlist(confidence_df[row, 'conf_R2'][[1]])[1]
    confidence_df[row, 'R3_OL_confidence'] <- NA
    confidence_df[row, 'R3_HS_confidence'] <- unlist(confidence_df[row, 'conf_R3'][[1]])[1]
  }
  
}


### Correctly match confidences in A&B and melt the c conditions 
key_cols <- c("batch_id", "tweet_in_batch", "tweet_id",
              "tweet", "condition", "shot",
              "R1_label", "R2_label", "R3_label",
              "R1_score", "R2_score", "R3_score",
              "cond_clean")

## OL / HS confidence column --> where is OL the first value --> A& C.OL
ol_first <- with(confidence_df,
                 str_detect(condition, "A")     |       # A
                   str_detect(condition, "C\\.OL"))     # C.OL


## Build AB confidences (straightforaward)
confidence_df_A_B <- confidence_df %>%
  filter(str_detect(condition, "[AB]")) %>%   # match A or B
  select(all_of(key_cols)) 

confidence_df_A_B <- merge(
  confidence_df_A_B,
  confidence_df[, c('tweet_id', 'cond_clean',
                    "R1_OL", "R2_OL", "R3_OL",
                    "R1_HS", "R2_HS", "R3_HS",
                    "conf_R1", "conf_R2", "conf_R3",
                    paste0("R", c(1,2,3), "_OL_confidence"),
                    paste0("R", c(1,2,3), "_HS_confidence"))],
  on = c('tweet_id', 'cond_clean'))

## Build C Block (always combine the C.HS and C.OL rwos per tweet x condition)
# C.OL
confidence_df_C.OL <- confidence_df %>%
  filter(grepl("C.OL", condition)) %>%              
  select(all_of(c(key_cols,
                  "R1_OL", "R2_OL", "R3_OL",
                  "conf_R1", "conf_R2", "conf_R3",
                  paste0("R", c(1,2,3), "_OL_confidence")))) 
colnames(confidence_df_C.OL)[which(colnames(confidence_df_C.OL) %in% 
                                     c("conf_R1", "conf_R2", "conf_R3"))] <- 
  c(paste0(c("conf_R1", "conf_R2", "conf_R3"),'_OL'))
  
# C.HS
confidence_df_C.HS <- confidence_df %>%
  filter(grepl("C.HS", condition)) %>%               
  select(all_of(c(key_cols,
                  "R1_HS", "R2_HS", "R3_HS",
                  "conf_R1", "conf_R2", "conf_R3",
                  paste0("R", c(1,2,3), "_HS_confidence")))) 
colnames(confidence_df_C.HS)[which(colnames(confidence_df_C.HS) %in% 
                                     c("conf_R1", "conf_R2", "conf_R3"))] <- 
  c(paste0(c("conf_R1", "conf_R2", "conf_R3"),'_HS'))


confidence_df_C <- merge(
  confidence_df_C.OL, 
  confidence_df_C.HS[,c('tweet_id', 'cond_clean',
                        "R1_HS", "R2_HS", "R3_HS",
                        "conf_R1_HS", "conf_R2_HS", "conf_R3_HS",
                        paste0("R", c(1,2,3), "_HS_confidence"))],
  on = c('tweet_id', 'cond_clean'))



## Combine final confidence df
confidence_df_final <- bind_rows(confidence_df_A_B, confidence_df_C)
confidence_df_final$R1_OL_confidence <- as.numeric(confidence_df_final$R1_OL_confidence)
confidence_df_final$R2_OL_confidence <- as.numeric(confidence_df_final$R2_OL_confidence)
confidence_df_final$R3_OL_confidence <- as.numeric(confidence_df_final$R3_OL_confidence)
confidence_df_final$R1_HS_confidence <- as.numeric(confidence_df_final$R1_HS_confidence)
confidence_df_final$R2_HS_confidence <- as.numeric(confidence_df_final$R2_HS_confidence)
confidence_df_final$R3_HS_confidence <- as.numeric(confidence_df_final$R3_HS_confidence)

## sanity checks
# nrow(confidence_df_A_B)
# nrow(confidence_df_C)
# nrow(confidence_df_final)


###############
######## Plot the average certainties per condition 
confidence_df_final['mean_tweet_conf_OL'] <- 
  rowMeans(confidence_df_final[, paste0('R', c(1,2,3), '_OL_confidence')])
confidence_df_final['mean_tweet_conf_HS'] <- 
  rowMeans(confidence_df_final[, paste0('R', c(1,2,3), '_HS_confidence')])


mean_confidence_wide <- confidence_df_final %>%
  group_by(cond_clean) %>%
  summarise(
    mean_confidence_OL = mean(mean_tweet_conf_OL, na.rm = TRUE),
    mean_confidence_HS = mean(mean_tweet_conf_HS, na.rm = TRUE),
    .groups = "drop"
  )

# Define color palette
palette <- c("OL" = "#1f77b4", "HS" = "#ff7f0e")

# Separate normal and batch conditions
normal_conditions <- mean_confidence_wide %>%
  filter(str_detect(cond_clean, "(?<!_batch)_conf$"))

batch_conditions <- mean_confidence_wide %>%
  filter(str_detect(cond_clean, "_batch_conf"))

# Sort conditions alphabetically (A,B,C)
normal_conditions <- normal_conditions %>% arrange(cond_clean)
batch_conditions <- batch_conditions %>% arrange(cond_clean)

# Concatenate for ordered display
ordered_df <- bind_rows(normal_conditions, batch_conditions)

# wide to long
long_df <- ordered_df %>%
  pivot_longer(
    cols = c(mean_confidence_OL, mean_confidence_HS),
    names_to = "group",
    values_to = "mean_confidence"
  ) %>%
  mutate(
    group = str_replace(group, "mean_confidence_", ""),
    group = factor(group, levels = c("OL", "HS")),
    cond_clean = factor(cond_clean, levels = ordered_df$cond_clean)
  )

# Plot
ggplot(long_df, aes(x = cond_clean, y = mean_confidence, fill = group)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ group, ncol = 2, scales = "fixed") +
  scale_fill_manual(values = palette) +
  theme_minimal(base_size = 16) +
  theme(
    axis.title.x  = element_text(size = 18, face = "bold"),
    axis.title.y  = element_text(size = 18, face = "bold"),
    axis.text.x   = element_text(angle = 45, hjust = 1, size = 14),
    axis.text.y   = element_text(size = 14),
    strip.text    = element_text(size = 16, face = "bold"),
    plot.margin   = margin(10, 10, 10, 10)
  ) +
  labs(
    x = NULL,
    y = "Mean LLM-reported Confidence"
    # title = "Confidence by Condition and Group"
  ) +
  coord_cartesian(ylim = c(70, 100))


ggsave("Plots/certainty_selfreport.pdf",
       last_plot(),
       device = cairo_pdf,
       width  = 20,            # close to dina4
       height = 13,
       units  = "cm"
)


################################################################################
###' *CERTAINTY METRIC 2: (EMPIRICAL) ENTROPY*
### --> for all conditions
################################################################################

key_cols <- c("batch_id", "tweet_in_batch", "tweet_id",
              "tweet", "condition", "shot",
              "cond_clean")

entropy_df <- all_data

entropy_df <- entropy_df %>%
  mutate(
    R1_OL = as.integer(str_detect(R1_label, "(?<!N)OL")),
    R2_OL = as.integer(str_detect(R2_label, "(?<!N)OL")),
    R3_OL = as.integer(str_detect(R3_label, "(?<!N)OL")),
    R1_HS = as.integer(str_detect(R1_label, "(?<!N)HS")),
    R2_HS = as.integer(str_detect(R2_label, "(?<!N)HS")),
    R3_HS = as.integer(str_detect(R3_label, "(?<!N)HS")),
  )



## combine A&B and C conditions again
entropy_df_A_B <- entropy_df %>%
  filter(str_detect(condition, "[AB]")) %>%                
  select(all_of(key_cols)) 

entropy_df_A_B <- merge(
  entropy_df_A_B,
  entropy_df[, c('tweet_id', 'cond_clean',
                  "R1_OL", "R2_OL", "R3_OL",
                  "R1_HS", "R2_HS", "R3_HS")],
  on = c('tweet_id', 'cond_clean'))



## Buil the C data frame
# C.OL
entropy_df_C.OL <- entropy_df %>%
  filter(grepl("C.OL", condition)) %>%                
  select(all_of(c(key_cols,
                  "R1_OL", "R2_OL", "R3_OL")))

# C.HS
entropy_df_C.HS <- entropy_df %>%
  filter(grepl("C.HS", condition)) %>%                
  select(all_of(c(key_cols,
                  "R1_HS", "R2_HS", "R3_HS")))


entropy_df_C <- merge(
  entropy_df_C.OL, 
  entropy_df_C.HS[,c('tweet_id', 'cond_clean',
                        "R1_HS", "R2_HS", "R3_HS")],
  on = c('tweet_id', 'cond_clean'))



entropy_df_full <- rbind(entropy_df_A_B, entropy_df_C)


# Function for binary shannon entropy
shannon_entropy <- function(p) {
  # basic formula (with base 2)
  entropy <- -p * log2(p) - (1 - p) * log2(1 - p)
  # handle NaN case (i.e., entropy if 111 or 000)
  entropy[is.nan(entropy)] <- 0  
  return(entropy) # entropy unit are shannons (base 2)
}

# Compute proportion (p) of 1s per OL/HS
entropy_df_full$p_OL <- 
  rowMeans(entropy_df_full[, paste0('R', c(1,2,3), '_OL')])
entropy_df_full$p_HS <- 
  rowMeans(entropy_df_full[, paste0('R', c(1,2,3), '_HS')])

# calculate entropy
entropy_df_full$tweet_entropy_OL <- shannon_entropy(entropy_df_full$p_OL)
entropy_df_full$tweet_entropy_HS <- shannon_entropy(entropy_df_full$p_HS)

# avg entropies per cond x outcome
mean_entropy_wide <- entropy_df_full %>%
  group_by(cond_clean) %>%
  summarise(
    mean_entropy_OL = mean(tweet_entropy_OL, na.rm = TRUE),
    mean_entropy_HS = mean(tweet_entropy_HS, na.rm = TRUE),
    .groups = "drop"
  )


# order for plotting
condition_order <- c(
  'A', 'A_conf', 'A_batch', 'A_batch_conf',
  'B', 'B_conf', 'B_batch', 'B_batch_conf',
  'C', 'C_conf', 'C_batch', 'C_batch_conf'
)

# cond_ckean order
mean_entropy_wide <- mean_entropy_wide %>%
  mutate(cond_clean = factor(cond_clean, levels = condition_order))

# wide to long
mean_entropy_long <- mean_entropy_wide %>%
  pivot_longer(
    cols = c(mean_entropy_OL, mean_entropy_HS),
    names_to = "group",
    names_prefix = "mean_entropy_",
    values_to = "mean_entropy"
  ) %>%
  mutate(group = factor(group, levels = c("OL", "HS")))

# Plot
palette <- c("OL" = "#1f77b4", "HS" = "#ff7f0e")

ggplot(mean_entropy_long, 
       aes(x = cond_clean, y = mean_entropy, fill = group)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ group, ncol = 2, scales = "fixed") +
  scale_fill_manual(values = palette) +
  theme_minimal(base_size = 16) +
  theme(
    axis.title.x  = element_text(size = 18, face = "bold"),
    axis.title.y  = element_text(size = 18, face = "bold"),
    axis.text.x   = element_text(angle = 45, hjust = 1, size = 14),
    axis.text.y   = element_text(size = 14),
    strip.text    = element_text(size = 16, face = "bold"),
    plot.margin   = margin(10, 10, 10, 10)
  ) +
  labs(
    x = NULL,
    y = "Mean entropy",
    # title = "Mean Entropy by Condition and Group"
  )



ggsave("Plots/certainty_entropy.pdf",
       last_plot(),
       device = cairo_pdf,
       width  = 20,            # close to dina4
       height = 13,
       units  = "cm"
)



################################################################################
###' *CERTAINTY METRIC 3: TOP2-TOKEN-LOGPROB DISPARITY*
### --> only non batch conditions
################################################################################

logprob_conditions <- c('A', 'B', 'C.OL', 'C.HS',
                        paste0(c('A', 'B', 'C.OL', 'C.HS'), '_conf')
                        )

logprob_df <- all_data %>% 
  filter(condition %in% logprob_conditions) %>%
  filter(shot == 'few-shot')


logprob_df <- logprob_df %>%
  mutate(
    R1_OL = as.integer(str_detect(R1_label, "(?<!N)OL")),
    R2_OL = as.integer(str_detect(R2_label, "(?<!N)OL")),
    R3_OL = as.integer(str_detect(R3_label, "(?<!N)OL")),
    R1_HS = as.integer(str_detect(R1_label, "(?<!N)HS")),
    R2_HS = as.integer(str_detect(R2_label, "(?<!N)HS")),
    R3_HS = as.integer(str_detect(R3_label, "(?<!N)HS")),
  )



# logprob df
ID_VARS <- c("batch_id", "tweet_in_batch", "tweet_id",
             "tweet", "condition", "shot", 'cond_clean', 
             paste0('R', c(1,2,3), '_label'),
             paste0('R', c(1,2,3), '_score'),
             paste0('R1_Token', c(1,2,3,4,5)),
             paste0('R2_Token', c(1,2,3,4,5)),
             paste0('R3_Token', c(1,2,3,4,5)))

raters <- c("R1", "R2", "R3")

# get long df witht the different rows for raters
long_df <- map_dfr(raters, function(r) {
  logprob_df %>%
    select(all_of(ID_VARS)) %>%            # keep identifier columns
    mutate(
      Rater  = r,
      label  = .data[[paste0(r, "_label")]],
      score  = .data[[paste0(r, "_score")]],
      Token1 = .data[[paste0(r, "_Token1")]],
      Token2 = .data[[paste0(r, "_Token2")]],
      Token3 = .data[[paste0(r, "_Token3")]],
      Token4 = .data[[paste0(r, "_Token4")]],
      Token5 = .data[[paste0(r, "_Token5")]]
    )
})


long_df <- long_df %>%
  mutate(across(starts_with("Token"), ~ as.character(.))) %>%
  mutate(
    # OL: get the OL tokens from the respective position per condition
    OL_top5_logprobs = case_when(
      condition %in% c("A", "A_conf", 
                       "C.OL", "C.OL_conf") ~ Token1, # Ol first
      condition == "B"                      ~ Token3, # OL sec
      condition == "B_conf"                 ~ Token4, # Ol sec
      TRUE                                  ~ NA
    ),
    
    # same for HS
    HS_top5_logprobs = case_when(
      condition == "A"                      ~ Token3, # HS second
      condition == "A_conf"                 ~ Token4, # HS second w/ score 
      condition %in% c("B", "B_conf",
                       "C.HS", "C.HS_conf") ~ Token1, # HS first
      TRUE                                  ~ NA
    )
  )



## Turn the messy logprob cell into ordered list
LOGPAT <- regex(
  "^(?<tok>.*?)\\s+(?<logp>[-+]?(?:\\d*\\.\\d+|\\d+)(?:e[-+]?\\d+)?)\\s*$",
  # --> matches: tok (= token) --> the character token
  #              logp (= logprob) --> the number afterwards
  ignore_case = TRUE
)

# function using the regex
to_list <- function(cell) {
  # Silently return empty list on NA / non-string inputs
  if (!is.character(cell) || is.na(cell)) return(list())
  
  lines <- str_split(cell, "\n", simplify = TRUE)
  
  map(lines, function(line) {
    m <- str_match(line, LOGPAT)
    if (!is.na(m[1, 1])) {
      # Named capture groups "tok" and "logp"
      list(token = m[1, "tok"],
           logp  = as.numeric(m[1, "logp"]))
    } else {
      NULL
    }
  }) %>% compact()            # drop NULLs
}

# use function
long_df <- long_df %>%
  mutate(
    OL_top5_list = map(OL_top5_logprobs, to_list),
    HS_top5_list = map(HS_top5_logprobs, to_list)
  )


# extract the first logprob for each logprob list 
first_logp <- function(x) {
  # x is one element of OL_top5_list / HS_top5_list 
  if (length(x) >= 1 && !is.null(x[[1]]$logp)) {
    x[[1]]$logp          
  } else {
    NaN
  }
}

# get logprobs for the first token
long_df <- long_df %>%
  mutate(
    # 1. grab first log-prob (or NA) ----
    OL_first_token_logprob = map_dbl(OL_top5_list, first_logp),
    HS_first_token_logprob = map_dbl(HS_top5_list, first_logp),
    
    # 2. convert to probability ---------
    OL_first_token_prob    = exp(OL_first_token_logprob),
    HS_first_token_prob    = exp(HS_first_token_logprob)
  )


### Plot Probs and Logprobs --> not used in the manuscript
# same colours as your Matplotlib palette
pal <- c(OL = "#1f77b4", HS = "#ff7f0e")

p1 <- ggplot(long_df, aes(OL_first_token_logprob)) +
  geom_histogram(bins = 50, fill = pal["OL"]) +
  labs(title = "OL – log p", x = "log-probability", y = "Count") +
  theme_minimal()

p2 <- ggplot(long_df, aes(HS_first_token_logprob)) +
  geom_histogram(bins = 50, fill = pal["HS"]) +
  labs(title = "HS – log p", x = "log-probability", y = "Count") +
  theme_minimal()

p3 <- ggplot(long_df, aes(OL_first_token_prob)) +
  geom_histogram(bins = 50, fill = pal["OL"]) +
  labs(title = "OL – p", x = "probability", y = "Count") +
  theme_minimal()

p4 <- ggplot(long_df, aes(HS_first_token_prob)) +
  geom_histogram(bins = 50, fill = pal["HS"]) +
  labs(title = "HS – p", x = "probability", y = "Count") +
  theme_minimal()

# patchwork plot
(p1 | p2) / (p3 | p4)



####' *Calculate Top2-Token Disparity Ratings*
# get first and second token logprob
first_logp  <- function(x) {
  if (length(x) >= 1 && !is.null(x[[1]]$logp)) x[[1]]$logp else NA_real_
}
second_logp <- function(x) {
  if (length(x) >= 2 && !is.null(x[[2]]$logp)) x[[2]]$logp else NA_real_
}

# get the logprobs in the df
long_df <- long_df %>%
  mutate(
    # first LOGPROBS
    OL_first_token_logprob = map_dbl(OL_top5_list, first_logp),
    HS_first_token_logprob = map_dbl(HS_top5_list, first_logp),
    
    # second LOGPROBS
    OL_second_token_logprob = map_dbl(OL_top5_list, second_logp),
    HS_second_token_logprob = map_dbl(HS_top5_list, second_logp),
    
    # get the PROBS
    OL_first_token_prob  = exp(OL_first_token_logprob),
    HS_first_token_prob  = exp(HS_first_token_logprob),
    OL_second_token_prob = exp(OL_second_token_logprob),
    HS_second_token_prob = exp(HS_second_token_logprob)
  )


# top 2 Token Disparity log(1) - log(2)
long_df$top2_token_disparity_OL <- 
  long_df$OL_first_token_logprob - long_df$OL_second_token_logprob
long_df$top2_token_disparity_HS <- 
  long_df$HS_first_token_logprob - long_df$HS_second_token_logprob


# Combine the C disparities
disparity_df <- long_df %>%
  select(
    batch_id, tweet_in_batch, tweet_id, tweet,
    condition, cond_clean, Rater, shot, label,
    OL_top5_list, HS_top5_list,
    top2_token_disparity_OL, top2_token_disparity_HS
  ) 

key_cols <- c("batch_id", "tweet_in_batch", "tweet_id",
              "tweet", "condition", "shot", "Rater", "cond_clean")

# AB
disparity_df_A_B <- disparity_df %>%
  filter(str_detect(condition, "[AB]"))   

# C conds merge 
disparity_df_C <- disparity_df %>%
  filter(grepl('C', condition)) %>%
  select(all_of(key_cols)) %>%
  distinct(tweet_id, cond_clean, Rater, .keep_all = TRUE)

# C.OL
C_OL <- disparity_df %>%
  filter(str_detect(condition, "C\\.OL")) %>%
  select(tweet, tweet_id, cond_clean, Rater,
         OL_top5_list, top2_token_disparity_OL)

disparity_df_C <- disparity_df_C %>%
  left_join(C_OL, by = c("tweet","tweet_id", "cond_clean", "Rater"))

# -C.HS
C_HS <- disparity_df %>%
  filter(str_detect(condition, "C\\.HS")) %>%
  select(tweet, tweet_id, cond_clean, Rater,
         HS_top5_list, top2_token_disparity_HS)

disparity_df_C <- disparity_df_C %>%
  left_join(C_HS, by = c("tweet","tweet_id", "cond_clean", "Rater"))

# full df with disparities
disparity_df <- bind_rows(disparity_df_A_B, disparity_df_C)



#### Plot Top2 Token Disparities
# get mean top2 token disparities
stats <- disparity_df %>%         
  group_by(cond_clean) %>% 
  summarise(
    across(
      c(top2_token_disparity_OL, top2_token_disparity_HS),
      list(
        mean = ~ mean(.x, na.rm = TRUE)
        # ,sem  = ~ sd(.x,  na.rm = TRUE) / sqrt(sum(!is.na(.x)))
      ),
      .names = "{.col}_{.fn}"
    ),
    .groups = "drop"
  )

# reshape
plot_df <- stats %>%
  pivot_longer(
    -cond_clean,
    names_to   = c("Token", ".value"),   # get mean col
    names_pattern = "(OL|HS)_(.*)"       
  ) # %>% mutate(ci95 = 1.96 * sem)   # 95ci --> not included 

# reorder
cond_levels <- c("A", "B", "C", "A_conf", "B_conf", "C_conf")
plot_df <- plot_df %>%
  mutate(
    Token      = factor(Token,      levels = c("OL", "HS")),
    cond_clean = factor(cond_clean, levels = cond_levels)
  )


# plot
pal <- c(OL = "#1f77b4", HS = "#ff7f0e")   

ggplot(plot_df,
       aes(x = cond_clean, y = mean, fill = cond_clean)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.7) +
  # geom_errorbar(
  #   aes(ymin = mean - ci95, ymax = mean + ci95),
  #   position = position_dodge(width = 0.8),
  #   width = 0.25
  # ) +
  scale_y_continuous(breaks = seq(0, 16, 2), limits = c(0, 17)) +
  scale_fill_manual(values = pal) +
  labs(
    x = "Condition",
    y = "Average Top-2 Token Disparity",
    title = "Average Top-2 Token Disparities by Condition (95 % CI)",
    fill = NULL
  ) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

ggplot(plot_df,
       aes(x = cond_clean, y = mean, fill = Token)) +
  geom_col(show.legend = FALSE) +                     # no legend
  facet_wrap(~ Token, ncol = 2, scales = "fixed") +   # OL panel then HS panel
  scale_fill_manual(values = pal) +                   # your custom palette per condition
  theme_minimal(base_size = 16) +                     # larger base text
  theme(
    axis.title.x  = element_text(size = 18, face = "bold"),
    axis.title.y  = element_text(size = 18, face = "bold"),
    axis.text.x   = element_text(angle = 45, hjust = 1, size = 14),
    axis.text.y   = element_text(size = 14),
    strip.text    = element_text(size = 16, face = "bold"),  # OL / HS titles
    plot.margin   = margin(10, 10, 10, 10)
  ) +
  labs(
    x = "Condition",
    y = "Average Top-2 Token Disparity"
  )


ggsave("Plots/certainty_disparity.pdf",
       last_plot(),
       device = cairo_pdf,
       width  = 20,            # close to dina4
       height = 13,
       units  = "cm"
)



################################################################################
###' *Agreement of the different uncertainty measures* (correlations)
################################################################################

# get the dfs all together
confidence_df_final <- confidence_df_final %>%
  mutate(
    mean_confidence_OL = rowMeans(
      select(., R1_OL_confidence, R2_OL_confidence, R3_OL_confidence),
      na.rm = TRUE
    ),
    mean_confidence_HS = rowMeans(
      select(., R1_HS_confidence, R2_HS_confidence, R3_HS_confidence),
      na.rm = TRUE
    )
  )

# tweet level mean disparity
disparity_df_means <- disparity_df %>% 
  group_by(cond_clean, tweet_id) %>% 
  summarise(
    mean_disparity_OL = mean(top2_token_disparity_OL, na.rm = TRUE),
    mean_disparity_HS = mean(top2_token_disparity_HS, na.rm = TRUE),
    .groups = "drop"
  )


### Plot correlations
# order
order_vars <- c("self_report", "entropy", "disparity")
rev_order  <- rev(order_vars)           

# Merge OL metrics
merged_ol <- entropy_df_full %>% 
  select(tweet_id, cond_clean, entropy = tweet_entropy_OL) %>% 
  left_join(
    confidence_df_final %>% 
      select(tweet_id, cond_clean, self_report = mean_tweet_conf_OL),
    by = c("tweet_id", "cond_clean")
  ) %>% 
  left_join(
    disparity_df_means %>% 
      select(tweet_id, cond_clean, disparity = mean_disparity_OL),
    by = c("tweet_id", "cond_clean")
  )

# Correlation & pvalues
ct_ol <- merged_ol %>% 
  select(all_of(order_vars)) %>% 
  psych::corr.test(use = "pairwise")

# tidy matrix
corr_ol_df <- ct_ol$r %>% 
  as.data.frame() %>% 
  rownames_to_column("Var1") %>% 
  pivot_longer(-Var1, names_to = "Var2", values_to = "r") %>% 
  left_join(
    ct_ol$p %>% 
      as.data.frame() %>% 
      rownames_to_column("Var1") %>% 
      pivot_longer(-Var1, names_to = "Var2", values_to = "p"),
    by = c("Var1", "Var2")
  ) %>% 
  mutate(
    stars = case_when(
      p < 0.001 ~ " ***",
      p < 0.01  ~ " **",
      p < 0.05  ~ " *",
      TRUE      ~ ""
    ),
    label = sprintf("%.2f%s", r, stars),
    Var2  = factor(Var2, levels = order_vars),   # x: left → right
    Var1  = factor(Var1, levels = rev_order)     # y: top → bottom
  )

# OL heatmap
pretty_labels <- c(
  "self_report" = "Self-\nReport",
  "entropy" = "Entropy",
  "disparity" = "Top2-\nToken-\nDisparity"
)

p_ol <- ggplot(corr_ol_df, aes(Var2, Var1, fill = r)) +
  geom_tile() +
  geom_text(aes(label = label), size = 5) +
  scale_fill_gradient2(
    limits   = c(-1, 1), midpoint = 0,
    low      = "#2166ac", mid = "white", high = "#b2182b",
    name     = "r"
  ) +
  scale_x_discrete(labels = pretty_labels) +
  scale_y_discrete(labels = pretty_labels) +
  labs(title = "OL", x = NULL, y = NULL) +
  theme_minimal(base_size = 16) +
  theme(
    axis.text.x  = element_text(angle = 0, hjust = 0.5, size = 14),
    axis.text.y  = element_text(size = 14),
    legend.title = element_text(size = 16, face = "bold"),
    legend.text  = element_text(size = 14),
    plot.title   = element_text(size = 18, face = "bold", hjust = 0.5),
    panel.grid   = element_blank(),
    plot.margin  = margin(10, 10, 10, 10)
  )

p_ol

### same for HS
merged_hs <- entropy_df_full %>% 
  select(tweet_id, cond_clean, entropy = tweet_entropy_HS) %>% 
  left_join(
    confidence_df_final %>% 
      select(tweet_id, cond_clean, self_report = mean_tweet_conf_HS),
    by = c("tweet_id", "cond_clean")
  ) %>% 
  left_join(
    disparity_df_means %>% 
      select(tweet_id, cond_clean, disparity = mean_disparity_HS),
    by = c("tweet_id", "cond_clean")
  )

# Correlation and p-values
ct_hs <- merged_hs %>% 
  select(all_of(order_vars)) %>% 
  psych::corr.test(use = "pairwise")

# HS martrix
corr_hs_df <- ct_hs$r %>% 
  as.data.frame() %>% 
  rownames_to_column("Var1") %>% 
  pivot_longer(-Var1, names_to = "Var2", values_to = "r") %>% 
  left_join(
    ct_hs$p %>% 
      as.data.frame() %>% 
      rownames_to_column("Var1") %>% 
      pivot_longer(-Var1, names_to = "Var2", values_to = "p"),
    by = c("Var1", "Var2")
  ) %>% 
  mutate(
    stars = case_when(
      p < 0.001 ~ " ***",
      p < 0.01  ~ " **",
      p < 0.05  ~ " *",
      TRUE      ~ ""
    ),
    label = sprintf("%.2f%s", r, stars),
    Var2  = factor(Var2, levels = order_vars),
    Var1  = factor(Var1, levels = rev_order)
  )

# Heat-map for HS
# without y axis labels as they are side by side
p_hs <- ggplot(corr_hs_df, aes(Var2, Var1, fill = r)) +
  geom_tile() +
  geom_text(aes(label = label), size = 5) +
  scale_fill_gradient2(
    limits   = c(-1, 1), midpoint = 0,
    low      = "#2166ac", mid = "white", high = "#b2182b",
    name     = "r"
  ) +
  scale_x_discrete(labels = pretty_labels) +
  scale_y_discrete(labels = pretty_labels) +
  labs(title = "HS", x = NULL, y = NULL) +
  theme_minimal(base_size = 16) +
  theme(
    axis.text.x  = element_text(angle = 0, hjust = 0.5, size = 14),
    # axis.text.y  = element_text(size = 14),
    axis.text.y  = element_blank(),
    legend.title = element_text(size = 16, face = "bold"),
    legend.text  = element_text(size = 14),
    plot.title   = element_text(size = 18, face = "bold", hjust = 0.5),
    panel.grid   = element_blank(),
    plot.margin  = margin(10, 10, 10, 10)
  )

p_hs

# patckwork side by side
(p_ol | p_hs) + plot_layout(guides = "collect") &
  theme(legend.position = "right")


ggsave("Plots/certainty_correlations.pdf",
       last_plot(),
       device = cairo_pdf,
       width  = 20,            # close to dina4
       height = 13,
       units  = "cm"
)




















