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
### Load, RBind, and restructure the GPT Labelled Tweets
################################################################################

# few shot labels
date_to_import_f <- "2025_07_01"
files_f <- list.files(
  path = "Data_Collection/OL_NH/",
  pattern = paste0("^", date_to_import_f),
  full.names = TRUE
)

all_data <- files_f %>%
  map_df(read_csv, .id = "source") %>%
  arrange(batch_id, tweet_in_batch) %>%
  mutate(shot = "few-shot") %>%
  select(1:4, shot, everything())


### Restructure to Long Format
shared_cols <- c("batch_id", "tweet_in_batch", "tweet_id",
                 "tweet", "condition", 'shot')


# loop over responder IDs
df_list <- list()
for (responder in c("R1", "R2", "R3")) {
  # use the shared cols
  df_temp <- all_data[ , shared_cols, drop = FALSE]
  # add responder col
  df_temp$responder <- responder
  # get labels by responder
  df_temp$label  <- all_data[[paste0(responder, "_label")]]
  # save
  df_list[[responder]] <- df_temp
}

# concat to long df
df_long <- do.call(rbind, df_list)
row.names(df_long) <- NULL   # tidy up row names


# get binary OL and HS cols
df_long$OL <- ifelse(grepl("(?<!N)OL", df_long$label, perl = TRUE), 1L, 0L)
df_long$HS <- ifelse(grepl("(?<!N)HS", df_long$label, perl = TRUE), 1L, 0L)




################################################################################
### PLOT LINEAR EFFECTS (ALL CONDITIONS) (LMM)
################################################################################

### get df for plotting
df_melt <- df_long %>%
  pivot_longer(cols = c(OL, HS),
               names_to = "Measure",
               values_to = "Label") %>%
  select(batch_id, condition, tweet_in_batch, shot, Measure, Label) %>%
  ## drop the wrong rows for C.OL and C.HS
  filter(!(str_detect(condition, "C\\.OL") & Measure == "HS") &
           !(str_detect(condition, "C\\.HS") & Measure == "OL")) %>%
  ## remove the .OL and .HS from condition
  mutate(condition = str_remove(condition, "\\.OL|\\.HS")) %>%
  ## rename plot facets from conditions (s.t. we get the base conidition)
  mutate(
    group_type = case_when(
      !str_detect(condition, "_")       ~ "base",
      str_ends(condition, "_batch_conf")~ "batch_conf",
      str_ends(condition, "_conf")      ~ "conf",
      str_ends(condition, "_batch")     ~ "batch",
      TRUE                              ~ "other"
    ),
    base_condition = str_split_fixed(condition, "_", 2)[,1],
    shot_base = paste0(base_condition)
  ) %>%
  mutate(
    group_type = factor(group_type,
                        levels = c("base","conf","batch","batch_conf"))
  )

row_order <- c('A','B','C')

df_melt <- df_melt %>%
  mutate(shot_base = factor(shot_base, levels = row_order))

### get the proportions for the plot
df_prop <- df_melt %>%
  group_by(shot, condition, tweet_in_batch, Measure) %>%
  summarise(Proportion = mean(Label, na.rm = TRUE), .groups = "drop") %>%
  mutate(
    group_type = case_when(
      !str_detect(condition, "_")       ~ "base",
      str_ends(condition, "_batch_conf")~ "batch_conf",
      str_ends(condition, "_conf")      ~ "conf",
      str_ends(condition, "_batch")     ~ "batch",
      TRUE                              ~ "other"
    ),
    base_condition = str_split_fixed(condition, "_", 2)[,1],
    shot_base = factor(paste0(base_condition),
                       levels = row_order),
    group_type = factor(group_type,
                        levels = c("base","conf","batch","batch_conf"))
  )


### plot geom smooth slopes
palette <- c(OL = "#1f77b4", HS = "#ff7f0e")

g <- ggplot(df_prop,
            aes(tweet_in_batch, Proportion, 
                colour = Measure,
                fill = Measure)) +
  geom_point(size = 1.2) +
  geom_smooth(method = "glm", formula = y ~ x, se = TRUE) +
  scale_x_continuous(breaks = 1:6,
                     limits = c(1,6)) +
  scale_colour_manual(values = palette, name = NULL) +
  scale_fill_manual(values = palette, guide = "none") + 
  facet_grid(rows = vars(shot_base), cols = vars(group_type)) +
  labs(x = "Tweet Position in Batch",
       y = "Proportion") +
  theme_bw(base_size = 14) +
  theme(
    # make facet labels bold
    strip.text      = element_text(face = "bold"),
    # optionally bump axis titles & text further
    axis.title      = element_text(size = 16),
    axis.text       = element_text(size = 14),
    legend.text     = element_text(size = 14),
    legend.title    = element_text(size = 16)
  )

g


### Fit the GLMMs to get th pvals
# Label ~ tweet_in_batch + (1 | batch_id)

combos <- df_melt %>%
  distinct(shot_base, group_type, Measure)

pvals <- combos %>%
  mutate(p_value = pmap_dbl(list(shot_base, group_type, Measure),
                            function(sb, gt, meas) {
                              sub <- df_melt %>%
                                filter(shot_base == sb, 
                                       group_type == gt,
                                       Measure == meas)
                              
                              # skip if not enough variation
                              if(n_distinct(sub$batch_id) < 2 ||
                                 length(unique(sub$Label)) < 2) return(1)
                              
                              fit <- tryCatch(
                                glmer(Label ~ tweet_in_batch + (1 | batch_id),
                                      data = sub, family = binomial,
                                      control = glmerControl(
                                        optimizer = "bobyqa")),
                                error = function(e) NULL
                              )
                              if(is.null(fit)) return(1)
                              
                              p <- 
                                summary(fit)$coefficients["tweet_in_batch", 
                                                          "Pr(>|z|)"]
                              p
                            }))

### add borders to the significant panes
sig_panels <- pvals %>%
  group_by(shot_base, group_type) %>%
  summarise(sig = any(p_value < 0.05/24), .groups = "drop") %>%
  filter(sig)

if(nrow(sig_panels)) {
  g <- g +
    geom_rect(data = sig_panels,
              inherit.aes = FALSE,
              xmin = -Inf, xmax = Inf, ymin = -Inf, ymax = Inf,
              fill = "transparent",
              color = "black",
              linewidth = 2) +
    theme(panel.spacing = unit(0.6, "lines"))
}
g




### add asterisks for the significant slopes
asterisks <- pvals %>%
  filter(p_value < 0.05/24) %>%
  mutate(y = 0.9,
         hjust = c(2.5, 1.5)[match(Measure, c("OL","HS"))],
         label = "*")

g <- g +
  geom_text(data = asterisks,
            inherit.aes = FALSE,
            aes(x = Inf, y = y, label = label, 
                colour = Measure, hjust = hjust), 
            vjust = 0.6, 
            size = 10,
            fontface = "bold",
            show.legend = FALSE) +
  scale_colour_manual(values = palette, name = NULL) +
  scale_fill_manual(values = palette, guide = "none") # + 
  # ggtitle("Effect of Position on OL/HS Proportions 
  #         (asterisks = GLMM slope p < 0.05/24; random intercept = batch_id)")
  
g



ggsave("Plots/R/order_effects_linear_all.pdf", 
       last_plot(),
       device = cairo_pdf,     
       width  = 29,            # close to dina4
       height = 13,             
       units  = "cm"
)






################################################################################
### PLOT LINEAR EFFECTS (BATCH CONDITIONS ONLY) (LMM)
################################################################################

### subset batch data
df_melt_batch <- df_melt %>% filter(grepl('batch', condition))
df_prop_batch <- df_prop %>% filter(grepl('batch', condition))
pvals_batch <- pvals %>% filter(grepl('batch', group_type))

### plot geom smooth slopes
palette <- c(OL = "#1f77b4", HS = "#ff7f0e")

g <- ggplot(df_prop_batch,
            aes(tweet_in_batch, Proportion, 
                colour = Measure,
                fill = Measure)) +
  geom_point(size = 1.2) +
  geom_smooth(method = "glm", formula = y ~ x, se = TRUE) +
  scale_colour_manual(values = palette, name = NULL) +
  scale_fill_manual(values = palette, guide = "none") + 
  facet_grid(rows = vars(shot_base), cols = vars(group_type)) +
  labs(x = "Tweet Position in Batch",
       y = "Proportion") +
  theme_bw()

g

### sahde facets
sig_panels_batch <- pvals_batch %>%
  group_by(shot_base, group_type) %>%
  summarise(sig = any(p_value < 0.05/12), .groups = "drop") %>%
  filter(sig)

if(nrow(sig_panels_batch)) {
  g <- g +
    geom_rect(data = sig_panels,
              inherit.aes = FALSE,
              xmin = -Inf, xmax = Inf, ymin = -Inf, ymax = Inf,
              fill = "transparent",
              color = "black",
              linewidth = 2) +
    theme(panel.spacing = unit(0.6, "lines"))
}
g

### add asterisks
asterisks_batch <- pvals_batch %>%
  filter(p_value < 0.05/12) %>%
  mutate(y = 0.9,
         hjust = c(2.5, 1.5)[match(Measure, c("OL","HS"))],
         label = "*")

g <- g +
  geom_text(data = asterisks_batch,
            inherit.aes = FALSE,
            aes(x = Inf, y = y, label = label, 
                colour = Measure, hjust = hjust),
            vjust = 0.6, size = 10,
            fontface = "bold",
            show.legend = FALSE) +
  scale_colour_manual(values = palette, name = NULL) +
  scale_fill_manual(values = palette, guide = "none") 

g


# save
ggsave("Plots/R/order_effects_linear_batch_only.pdf", 
       last_plot(),
       device = cairo_pdf,     
       width  = 29,            
       height = 13,             
       units  = "cm"
)





################################################################################
### PLOT QUADRATIC EFFECTS (ALL CONDITIONS)
################################################################################

palette <- c(OL = "#1f77b4", HS = "#ff7f0e")

g <- ggplot(df_prop,
            aes(tweet_in_batch, Proportion, 
                colour = Measure,
                fill = Measure)) +
  geom_point(size = 1.2) +
  geom_smooth(method = "lm", formula = y ~ x + I(x^2), se = TRUE) +
  scale_x_continuous(breaks = 1:6,
                     limits = c(1,6)) +
  scale_colour_manual(values = palette, name = NULL) +
  scale_fill_manual(values = palette, guide = "none") + 
  facet_grid(rows = vars(shot_base), cols = vars(group_type)) +
  labs(x = "Tweet Position in Batch",
       y = "Proportion") +
    theme_bw(base_size = 14) +
  theme(
    # make facet labels bold
    strip.text      = element_text(face = "bold"),
    # optionally bump axis titles & text further
    axis.title      = element_text(size = 16),
    axis.text       = element_text(size = 14),
    legend.text     = element_text(size = 14),
    legend.title    = element_text(size = 16)
  )

g

### fit glmm with quadr fixed effects
combos <- df_melt %>%
  distinct(shot_base, group_type, Measure)

pvals <- combos %>%
  mutate(p_value = pmap_dbl(list(shot_base, group_type, Measure),
                            function(sb, gt, meas) {
                              sub <- df_melt %>%
                                filter(shot_base == sb, group_type == gt, Measure == meas)
                              
                              # skip if not enough variation
                              if(n_distinct(sub$batch_id) < 2 ||
                                 length(unique(sub$Label)) < 2) return(1)
                              
                              fit <- tryCatch(
                                glmer(Label ~ tweet_in_batch + I(tweet_in_batch^2) + (1 | batch_id),
                                      data = sub, family = binomial,
                                      control = glmerControl(optimizer = "bobyqa")),
                                error = function(e) NULL
                              )
                              if(is.null(fit)) return(1)
                              
                              p  <- summary(fit)$coefficients["I(tweet_in_batch^2)", "Pr(>|z|)"]
                              p
                            }))

pvals

### add borderlines where sig
sig_panels <- pvals %>%
  group_by(shot_base, group_type) %>%
  summarise(sig = any(p_value < 0.05 / 24), .groups = "drop") %>%
  filter(sig)

if(nrow(sig_panels)) {
  g <- g +
    geom_rect(data = sig_panels,
              inherit.aes = FALSE,
              xmin = -Inf, xmax = Inf, ymin = -Inf, ymax = Inf,
              fill = "transparent", 
              color = "black",
              linewidth = 2) +
    theme(panel.spacing = unit(0.6, "lines"))
}
g

### add asterisks
asterisks <- pvals %>%
  filter(p_value < 0.05/24) %>%
  mutate(y = 0.9,
         hjust = c(2.5, 1.5)[match(Measure, c("OL","HS"))],
         label = "*")

g <- g +
  geom_text(data = asterisks,
            inherit.aes = FALSE,
            aes(x = Inf, y = y, label = label, 
                colour = Measure,
                hjust = hjust),
            vjust = 0.3, size = 10,
            fontface = "bold",
            show.legend = FALSE) +
  scale_colour_manual(values = palette, name = NULL) 

g


# save
ggsave("Plots/R/order_effects_quadr_all.pdf", 
       last_plot(),
       device = cairo_pdf,     
       width  = 29,            # close to dina4
       height = 13,             
       units  = "cm"
)



################################################################################
### PLOT QUADRATIC EFFECTS (BATCH CONDITIONS ONLY)
################################################################################

# base plot
palette <- c(OL = "#1f77b4", HS = "#ff7f0e")

g <- ggplot(df_prop_batch,
            aes(tweet_in_batch, Proportion, 
                colour = Measure,
                fill = Measure)) +
  geom_point(size = 1.2) +
  geom_smooth(method = "lm", formula = y ~ x + I(x^2), se = TRUE) +
  scale_colour_manual(values = palette, name = NULL) +
  scale_fill_manual(values = palette, guide = "none") + 
  facet_grid(rows = vars(shot_base), cols = vars(group_type)) +
  labs(x = "Tweet Position in Batch",
       y = "Proportion") +
  theme_bw()
g

### fit glmm qith quadratic effects
combos <- df_melt_batch %>%
  distinct(shot_base, group_type, Measure)

pvals <- combos %>%
  mutate(p_value = pmap_dbl(list(shot_base, group_type, Measure),
                            function(sb, gt, meas) {
                              sub <- df_melt_batch %>%
                                filter(shot_base == sb, 
                                       group_type == gt, 
                                       Measure == meas)
                              
                              # skip if not enough variation
                              if(n_distinct(sub$batch_id) < 2 ||
                                 length(unique(sub$Label)) < 2) return(1)
                              
                              fit <- tryCatch(
                                glmer(Label ~ tweet_in_batch + I(tweet_in_batch^2) + (1 | batch_id),
                                      data = sub, family = binomial,
                                      control = glmerControl(optimizer = "bobyqa")),
                                error = function(e) NULL
                              )
                              if(is.null(fit)) return(1)
                              
                              p  <- summary(fit)$coefficients["I(tweet_in_batch^2)", "Pr(>|z|)"]
                              p
                            }))

pvals

### border lines
sig_panels <- pvals %>%
  group_by(shot_base, group_type) %>%
  summarise(sig = any(p_value < 0.05 / 12), .groups = "drop") %>%
  filter(sig)

if(nrow(sig_panels)) {
  g <- g +
    geom_rect(data = sig_panels,
              inherit.aes = FALSE,
              xmin = -Inf, xmax = Inf, ymin = -Inf, ymax = Inf,
              fill = "transparent", 
              color = "black",
              linewidth = 2) +
    theme(panel.spacing = unit(0.6, "lines"))
}
g

### asterisks
asterisks <- pvals %>%
  filter(p_value < 0.05/12) %>%
  mutate(y = c(0.90, 0.82)[match(Measure, c("OL","HS"))],
         label = "*")

g <- g +
  geom_text(data = asterisks,
            inherit.aes = FALSE,
            aes(x = Inf, y = y, label = label, colour = Measure),
            hjust = 1.5, vjust = -0.1, size = 10,
            fontface = "bold",
            show.legend = FALSE) +
  scale_colour_manual(values = palette, name = NULL) +
  ggtitle("Effect of Position on OL/HS Proportions (asterisks = GLMM slope p < 0.05/12; random intercept = batch_id)")

g


ggsave("Plots/R/order_effects_quadr_batch_only.pdf",
       last_plot(),
       device = cairo_pdf,     
       width  = 29,            # close to dina4
       height = 13,             
       units  = "cm"
)

