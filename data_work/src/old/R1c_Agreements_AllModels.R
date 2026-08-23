################################################################################
### run_agreement_analysis()
###
### Per-model agreement engine, generalizing the kappa-heatmap logic in
### R1_Prevalences_Agreements.R (there hardcoded to the original GPT-4o-mini
### OL_NH data) so it runs for any of the 7 models. All models are few-shot
### only, so (unlike the original script) there is no zero-shot branch here.
###
### Diagonal   = Fleiss' kappa across the 3 raters within a condition.
### Off-diag   = Cohen's kappa between conditions' modal ratings (upper
###              triangle only, to match the original heatmap).
### "reference_agreement" col = Cohen's kappa of each condition's modal
###              rating against the Kern et al. human-reference modal rating.
################################################################################

get_mode <- function(x) {
  x <- x[!is.na(x)]
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

run_agreement_analysis <- function(df_long, model_name, tweets_full) {

  library(tidyverse)
  library(stringr)
  library(irr)

  ### modal rating per tweet/condition (majority of the 3 raters)
  df_agg <- df_long %>%
    group_by(tweet_id, condition) %>%
    summarise(
      modal_HS = get_mode(HS),
      modal_OL = get_mode(OL),
      .groups = "drop"
    )

  conditions    <- sort(unique(df_long$condition))
  HS_conditions <- conditions[!grepl("C\\.OL", conditions)]
  OL_conditions <- conditions[!grepl("C\\.HS", conditions)]

  across_HS <- matrix(
    NaN,
    nrow = length(HS_conditions), ncol = length(HS_conditions),
    dimnames = list(HS_conditions, HS_conditions)
  )
  across_OL <- matrix(
    NaN,
    nrow = length(OL_conditions), ncol = length(OL_conditions),
    dimnames = list(OL_conditions, OL_conditions)
  )

  for (cond_i in conditions) {

    ### diagonal: Fleiss' kappa across the 3 raters within this condition
    df_sub <- df_long %>% filter(condition == cond_i)

    mat_OL <- df_sub %>%
      select(tweet_id, responder, OL) %>%
      pivot_wider(names_from = responder, values_from = OL) %>%
      select(-tweet_id) %>%
      as.matrix()

    mat_HS <- df_sub %>%
      select(tweet_id, responder, HS) %>%
      pivot_wider(names_from = responder, values_from = HS) %>%
      select(-tweet_id) %>%
      as.matrix()

    if (cond_i %in% colnames(across_HS)) {
      across_HS[cond_i, cond_i] <- irr::kappam.fleiss(mat_HS)$value
    }
    if (cond_i %in% colnames(across_OL)) {
      across_OL[cond_i, cond_i] <- irr::kappam.fleiss(mat_OL)$value
    }

    ### off-diagonal: Cohen's kappa between conditions' modal ratings
    for (cond_j in conditions) {
      if (cond_i == cond_j) next

      if (cond_i %in% OL_conditions && cond_j %in% OL_conditions) {
        wide_OL <- df_agg %>%
          filter(condition %in% OL_conditions) %>%
          select(tweet_id, condition, modal_OL) %>%
          pivot_wider(names_from = condition, values_from = modal_OL)
        pair_ol <- wide_OL %>% select(all_of(c(cond_i, cond_j))) %>% drop_na()
        across_OL[cond_i, cond_j] <- kappa2(pair_ol)$value
      }

      if (cond_i %in% HS_conditions && cond_j %in% HS_conditions) {
        wide_HS <- df_agg %>%
          filter(condition %in% HS_conditions) %>%
          select(tweet_id, condition, modal_HS) %>%
          pivot_wider(names_from = condition, values_from = modal_HS)
        pair <- wide_HS %>% select(all_of(c(cond_i, cond_j))) %>% drop_na()
        across_HS[cond_i, cond_j] <- kappa2(pair)$value
      }
    }
  }

  ### agreement of each condition's modal rating against the Kern et al.
  ### human-reference modal rating
  kern_modes <- tweets_full %>%
    group_by(tweet_id) %>%
    reframe(
      global_HS = get_mode(hate_speech),
      global_OL = get_mode(offensive_language)
    )

  df_agg2 <- df_agg %>% left_join(kern_modes, by = "tweet_id")

  outcomes <- c("HS", "OL")
  conds    <- unique(df_agg2$condition)

  kappa_results <- list()
  for (outcome in outcomes) {
    gcol <- paste0("global_", outcome)
    mcol <- paste0("modal_",  outcome)
    kappa_results[[outcome]] <- tibble(
      condition = conds,
      kappa = map_dbl(conds, function(cond) {
        tmp <- df_agg2 %>%
          filter(condition == cond) %>%
          select(all_of(c(mcol, gcol))) %>%
          drop_na()
        if (nrow(tmp) > 0) kappa2(tmp)$value else NA_real_
      })
    )
  }

  ### assemble long-format heatmap data (upper triangle only), one outcome
  ### panel at a time
  prefixes <- c("A", "B", "C")
  suffixes <- c("", "_conf", "_batch", "_batch_conf")
  base_cond_order <- paste0(rep(prefixes, each = length(suffixes)), suffixes)

  build_panel_df <- function(outcome = c("OL", "HS")) {

    outcome <- match.arg(outcome)
    mat     <- if (outcome == "OL") across_OL else across_HS

    ### drop the 'wrong' duplicated C.OL / C.HS rows-cols for this outcome
    other_outcome <- ifelse(outcome == "HS", "OL", "HS")
    no_cond       <- paste0("C.", other_outcome)
    keep_rows     <- !str_detect(rownames(mat), fixed(no_cond))
    keep_cols     <- !str_detect(colnames(mat), fixed(no_cond))
    mat           <- mat[keep_rows, keep_cols, drop = FALSE]

    strip_pattern <- "\\.(?:OL|HS)"
    rownames(mat) <- str_remove_all(rownames(mat), strip_pattern)
    colnames(mat) <- str_remove_all(colnames(mat), strip_pattern)

    cond_order <- base_cond_order[base_cond_order %in% rownames(mat)]
    mat        <- mat[cond_order, cond_order, drop = FALSE]
    mat_df     <- as.data.frame(mat)
    mat_df$gap <- NaN

    kappa_results_select <- kappa_results[[outcome]] %>%
      filter(!grepl(other_outcome, condition)) %>%
      mutate(condition = str_remove_all(condition, strip_pattern)) %>%
      filter(condition %in% cond_order) %>%
      select(condition, kappa) %>%
      deframe()
    kappa_select_df <- as.data.frame(kappa_results_select)
    colnames(kappa_select_df) <- "reference_agreement"

    mat_df <- merge(mat_df, kappa_select_df,
                     by.x = "row.names", by.y = "row.names", sort = FALSE)
    rownames(mat_df) <- mat_df$Row.names
    mat_df$Row.names  <- NULL

    df <- as_tibble(mat_df, rownames = "row_cond") |>
      pivot_longer(-row_cond, names_to = "col_cond", values_to = "value") |>
      mutate(outcome = factor(outcome, levels = c("OL", "HS")))

    cond_index <- setNames(seq_along(cond_order), cond_order)
    df <- df |>
      mutate(
        row_idx = cond_index[row_cond],
        col_idx = cond_index[col_cond],
        value   = ifelse(!is.na(row_idx) & !is.na(col_idx) &
                            col_idx > row_idx, NaN, value)
      ) |>
      select(-row_idx, -col_idx)

    df
  }

  heat_df <- map_dfr(c("OL", "HS"), build_panel_df)

  all_rows <- unique(heat_df$row_cond)
  all_cols <- unique(heat_df$col_cond)

  heat_df <- heat_df %>%
    mutate(
      row_cond   = factor(row_cond, levels = all_rows),
      col_cond   = factor(col_cond, levels = all_cols),
      model_name = model_name
    )

  list(
    across_OL     = across_OL,
    across_HS     = across_HS,
    kappa_results = kappa_results,
    heat_df       = heat_df
  )
}

################################################################################
### build_agreement_diag_table_split()
###
### Within-condition rater agreement (the diagonal of across_OL / across_HS,
### i.e. Fleiss' kappa across the 3 raters) as two separate wide tables --
### one for OL, one for HS -- each with one row per condition and one column
### per model. Split by outcome (rather than one table with OL/HS column
### pairs, as in build_wide_table()) so each half can be typeset as its own
### subtable.
################################################################################

extract_diagonal_df <- function(mat, outcome, model_name) {
  tibble(
    condition  = str_remove_all(names(diag(mat)), "\\.(?:OL|HS)"),
    kappa      = as.numeric(diag(mat)),
    outcome    = outcome,
    model_name = model_name
  )
}

condition_label_map <- c(
  A            = "Joint, OL first: Base",
  A_conf       = "Joint, OL first: + Conf.",
  A_batch      = "Joint, OL first: Batch",
  A_batch_conf = "Joint, OL first: Batch + Conf.",
  B            = "Joint, HS first: Base",
  B_conf       = "Joint, HS first: + Conf.",
  B_batch      = "Joint, HS first: Batch",
  B_batch_conf = "Joint, HS first: Batch + Conf.",
  C            = "Separate: Base",
  C_conf       = "Separate: + Conf.",
  C_batch      = "Separate: Batch",
  C_batch_conf = "Separate: Batch + Conf."
)

build_agreement_diag_table_split <- function(agreement_results, model_order, model_labels) {

  library(tidyverse)

  row_order <- unname(condition_label_map)
  labels_ordered <- model_labels[model_order]

  diag_long <- imap_dfr(agreement_results, function(res, key) {
    bind_rows(
      extract_diagonal_df(res$across_OL, "OL", key),
      extract_diagonal_df(res$across_HS, "HS", key)
    )
  }) %>%
    mutate(
      term_clean  = factor(condition_label_map[condition], levels = row_order),
      model_label = factor(model_labels[model_name], levels = labels_ordered),
      kappa_fmt   = sprintf("%.2f", kappa)
    )

  make_outcome_table <- function(oc) {
    diag_long %>%
      filter(outcome == oc) %>%
      select(term_clean, model_label, kappa_fmt) %>%
      pivot_wider(names_from = model_label, values_from = kappa_fmt) %>%
      arrange(term_clean) %>%
      select(term_clean, all_of(as.character(labels_ordered)))
  }

  list(OL = make_outcome_table("OL"), HS = make_outcome_table("HS"))
}

################################################################################
### agreement_diag_split_to_standalone_tex()
###
### Renders the OL / HS tables from build_agreement_diag_table_split() as
### two `subtable`s inside one `table` float (via the `subcaption` package),
### with an overall caption plus a methods note explaining the kappa and
### the condition labels.
################################################################################

agreement_diag_split_to_standalone_tex <- function(split_tables, model_labels) {

  colspec <- paste0("l", strrep("c", length(model_labels)))
  header  <- paste(c("Condition", model_labels), collapse = " & ")

  rows_for <- function(wide) {
    rows <- apply(wide, 1, function(r) {
      vals <- ifelse(is.na(r[model_labels]), "--", r[model_labels])
      paste(c(as.character(r[["term_clean"]]), vals), collapse = " & ")
    })
    paste(paste0(rows, " \\\\"), collapse = "\n")
  }

  make_subtable <- function(wide, caption, label) {
    c(
      "\\begin{subtable}{\\textwidth}",
      "\\centering",
      paste0("\\caption{", caption, "}"),
      paste0("\\label{", label, "}"),
      "\\resizebox{\\textwidth}{!}{%",
      paste0("\\begin{tabular}{", colspec, "}"),
      "\\toprule",
      paste0(header, " \\\\"),
      "\\midrule",
      rows_for(wide),
      "\\bottomrule",
      "\\end{tabular}%",
      "}",
      "\\end{subtable}"
    )
  }

  c(
    "\\documentclass{article}",
    "\\usepackage[margin=0.5in]{geometry}",
    "\\usepackage{booktabs}",
    "\\usepackage{graphicx}",
    "\\usepackage{caption}",
    "\\usepackage{subcaption}",
    "\\begin{document}",
    "\\begin{table}[t]",
    "\\centering",
    "\\caption{Within-condition rater agreement by task condition and model.}",
    "\\label{tab:agreement-within-condition}",
    make_subtable(split_tables$OL, "Offensive language (OL)", "tab:agreement-within-condition-ol"),
    "\\vspace{1em}",
    make_subtable(split_tables$HS, "Hate speech (HS)", "tab:agreement-within-condition-hs"),
    "\\smallskip",
    "\\begin{minipage}{\\textwidth}",
    "\\footnotesize",
    paste(
      "\\textit{Note}: Cell values are Fleiss' $\\kappa$ across the three ratings (R1--R3)",
      "collected for each tweet under a given condition, i.e. a model's agreement with itself",
      "under repeated prompting; they do not reflect agreement with the human reference labels.",
      "\\textit{Joint, OL first} / \\textit{Joint, HS first}: OL and HS are elicited from a single",
      "joint prompt, with the named outcome asked about first. \\textit{Separate}: OL and HS are",
      "elicited from two separate prompts. \\textit{+ Conf.}: the model additionally reports a",
      "confidence rating alongside the label. \\textit{Batch}: tweets are labeled in batches",
      "rather than one at a time."
    ),
    "\\end{minipage}",
    "\\end{table}",
    "\\end{document}"
  )
}
