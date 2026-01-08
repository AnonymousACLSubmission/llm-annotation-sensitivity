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
### PLOT LINEAR EFFECTS (BATCH CONDITIONS) (LPM)
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


### Fit the LPMs to get th pvals
# Label ~ tweet_in_batch + (1 | batch_id)

library(lmerTest)

combos <- df_melt %>%
  distinct(shot_base, group_type, Measure)

pvals <- combos %>%
  mutate(p_value = pmap_dbl(
    list(shot_base, group_type, Measure),
    function(sb, gt, meas) {
      
      sub <- df_melt %>%
        filter(
          shot_base == sb,
          group_type == gt,
          Measure == meas
        )
      
      # skip if not enough variation
      if (n_distinct(sub$batch_id) < 2 ||
          length(unique(sub$Label)) < 2) return(1)
      
      fit <- tryCatch(
        lmer(
          Label ~ tweet_in_batch + (1 | batch_id),
          data = sub,
          REML = FALSE
        ),
        error = function(e) NULL
      )
      
      if (is.null(fit)) return(1)
      
      # extract p-value for tweet_in_batch
      p <- summary(fit)$coefficients[
        "tweet_in_batch", "Pr(>|t|)"
      ]
      
      p
    }
  ))


### subset batch data
df_melt_batch <- df_melt %>% filter(grepl('batch', condition))
df_prop_batch <- df_prop %>% filter(grepl('batch', condition))
pvals_batch <- pvals %>% filter(grepl('batch', group_type))

### 1. Main Plot Construction
palette <- c(OL = "#1f77b4", HS = "#ff7f0e")

g <- ggplot(df_prop_batch,
            aes(tweet_in_batch, Proportion, 
                colour = Measure,
                fill = Measure)) +
  geom_point(size = 1.2) +
  geom_smooth(method = "glm", formula = y ~ x, se = TRUE) +
  scale_x_continuous(breaks = 1:6, limits = c(1, 6)) +
  scale_colour_manual(values = palette, name = NULL) +
  scale_fill_manual(values = palette, guide = "none") + 
  facet_grid(rows = vars(shot_base), cols = vars(group_type)) +
  labs(x = "Tweet Position in Batch",
       y = "Proportion") +
  # Match original theme and font sizes
  theme_bw(base_size = 14) +
  theme(
    strip.text         = element_text(face = "bold"),
    axis.title         = element_text(size = 16),
    axis.text          = element_text(size = 14),
    
    legend.position    = "bottom",
    legend.box         = "horizontal",
    
    legend.text        = element_text(size = 18),         
    
    legend.background  = element_rect(fill = "transparent"),
    legend.margin      = margin(t = 10),
    panel.grid.minor   = element_blank()
  )

### 2. Shade significant panels (Bonferroni alpha = 0.05/12 for batch subsets)
sig_panels_batch <- pvals_batch %>%
  group_by(shot_base, group_type) %>%
  summarise(sig = any(p_value < 0.05/12), .groups = "drop") %>%
  filter(sig)

if(nrow(sig_panels_batch)) {
  g <- g +
    geom_rect(data = sig_panels_batch, # Fixed variable name here
              inherit.aes = FALSE,
              xmin = -Inf, xmax = Inf, ymin = -Inf, ymax = Inf,
              fill = "transparent",
              color = "black",
              linewidth = 2) +
    theme(panel.spacing = unit(0.6, "lines"))
}
g

### 3. Add asterisks for significant slopes
asterisks_batch <- pvals_batch %>%
  filter(p_value < 0.05/12) %>%
  mutate(y = 0.85,
         # Adjusting match for only 2 batch columns
         hjust = c(2.5, 1.5)[match(Measure, c("OL","HS"))],
         label = "*")

g

g <- g +
  geom_text(data = asterisks_batch,
            inherit.aes = FALSE,
            aes(x = Inf, y = y, label = label, 
                colour = Measure, hjust = hjust),
            vjust = 0.6, 
            size = 12,
            fontface = "bold",
            show.legend = FALSE)

g

# save
ggsave("Plots/order_effects_linear_batch_only_lpm.pdf", 
       last_plot(),
       device = cairo_pdf,     
       width  = 14.5,            
       height = 13,             
       units  = "cm"
)






################################################################################
### PLOT QUADRATIC EFFECTS (BATCH CONDITIONS)
################################################################################

### fit LPM with quadr fixed effects
### Fit mixed-effects LPMs with quadratic term
# Label ~ tweet_in_batch + tweet_in_batch^2 + (1 | batch_id)

combos <- df_melt %>%
  distinct(shot_base, group_type, Measure)

pvals <- combos %>%
  mutate(p_value = pmap_dbl(
    list(shot_base, group_type, Measure),
    function(sb, gt, meas) {
      
      sub <- df_melt %>%
        filter(
          shot_base == sb,
          group_type == gt,
          Measure == meas
        )
      
      # skip if not enough variation
      if (n_distinct(sub$batch_id) < 2 ||
          length(unique(sub$Label)) < 2) return(1)
      
      fit <- tryCatch(
        lmer(
          Label ~ tweet_in_batch + I(tweet_in_batch^2) + (1 | batch_id),
          data = sub,
          REML = FALSE
        ),
        error = function(e) NULL
      )
      
      if (is.null(fit)) return(1)
      
      # p-value for quadratic term
      p <- summary(fit)$coefficients[
        "I(tweet_in_batch^2)", "Pr(>|t|)"
      ]
      
      p
    }
  ))



### 1. Filter Data for specific group types
# only 'batch' and 'batch_conf' are included
target_groups <- c("batch", "batch_conf")

df_plot_filtered <- df_prop_batch %>% 
  filter(group_type %in% target_groups)

pvals_filtered <- pvals %>% 
  filter(group_type %in% target_groups)

### 2. Main Quadratic Plot Construction
palette <- c(OL = "#1f77b4", HS = "#ff7f0e")

g <- ggplot(df_plot_filtered,
            aes(tweet_in_batch, Proportion, 
                colour = Measure,
                fill = Measure)) +
  geom_point(size = 1.2) +
  geom_smooth(method = "glm", formula = y ~ x + I(x^2), se = TRUE) +
  scale_x_continuous(breaks = 1:6, limits = c(1, 6)) +
  scale_colour_manual(values = palette, name = NULL) +
  scale_fill_manual(values = palette, guide = "none") + 
  facet_grid(rows = vars(shot_base), cols = vars(group_type)) +
  labs(x = "Tweet Position in Batch",
       y = "Proportion") +
  theme_bw(base_size = 14) +
  theme(
    strip.text         = element_text(face = "bold"),
    axis.title         = element_text(size = 16),
    axis.text          = element_text(size = 14),
    
    # Legend settings
    legend.position    = "bottom",
    legend.box         = "horizontal",
    legend.text        = element_text(size = 18),
    
    legend.background  = element_rect(fill = "transparent"),
    legend.margin      = margin(t = 10),
    panel.grid.minor   = element_blank()
  )

### 3. Shade significant panels
sig_panels_quad <- pvals_filtered %>%
  group_by(shot_base, group_type) %>%
  summarise(sig = any(p_value < 0.05 / 12, na.rm = TRUE), .groups = "drop") %>%
  filter(sig)

if(nrow(sig_panels_quad)) {
  g <- g +
    geom_rect(data = sig_panels_quad,
              inherit.aes = FALSE,
              xmin = -Inf, xmax = Inf, ymin = -Inf, ymax = Inf,
              fill = "transparent", 
              color = "black",
              linewidth = 1.5) +
    theme(panel.spacing = unit(0.6, "lines"))
}

### 4. Add asterisks SIDE-BY-SIDE
asterisks_quad <- pvals_filtered %>%
  filter(p_value < 0.05/12) %>%
  mutate(
    y = 0.9, 
    hjust = ifelse(Measure == "OL", 2.5, 1.5),
    label = "*"
  )

if(nrow(asterisks_quad) > 0) {
  g <- g +
    geom_text(data = asterisks_quad,
              inherit.aes = FALSE,
              aes(x = Inf, y = y, label = label, 
                  colour = Measure, hjust = hjust),
              vjust = 0.6, 
              size = 12, 
              fontface = "bold",
              show.legend = FALSE)
}

g


ggsave("Plots/order_effects_quadr_batch_only_lpm.pdf",
       last_plot(),
       device = cairo_pdf,     
       width  = 14.5,            # close to dina4
       height = 13,             
       units  = "cm"
)

