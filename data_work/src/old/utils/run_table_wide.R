################################################################################
### build_wide_table()
###
### Assembles a single wide Table 1 (one column-pair per model) from a named
### list of run_inference_analysis() results (see run_inference_analysis_v2.R).
################################################################################

build_wide_table <- function(lpm_results, model_order, model_labels) {

  library(tidyverse)

  row_order <- c(
    "Joint: OL first",
    "Joint: HS first",
    "Separate Labeling",
    "With Confidence (vs. Without)",
    "Batch Prompt (vs. Indiv. Prompts)",
    "$\\sigma^2$",
    "$\\tau_{00}$\\textsubscript{tweet\\_id}",
    "ICC",
    "$N$\\textsubscript{tweet\\_id}",
    "Observations",
    "Marginal $R^2$",
    "Conditional $R^2$"
  )

  per_model <- map2(model_order, model_labels, function(key, label) {
    lpm_results[[key]]$paper_table %>%
      transmute(
        term_clean,
        !!paste0(label, "_OL") := OL_estimate_fmt,
        !!paste0(label, "_HS") := HS_estimate_fmt
      )
  })

  wide <- reduce(per_model, full_join, by = "term_clean") %>%
    mutate(term_clean = factor(term_clean, levels = row_order)) %>%
    arrange(term_clean)

  wide
}

################################################################################
### wide_table_to_latex()
###
### Renders the wide table as LaTeX rows, chunked into groups of
### `models_per_block` model-columns for \resizebox / landscape blocks.
################################################################################

wide_table_to_latex <- function(wide, model_labels, models_per_block = length(model_labels)) {

  library(tidyverse)

  blocks <- split(model_labels, ceiling(seq_along(model_labels) / models_per_block))

  map_chr(blocks, function(labels_block) {
    cols <- as.vector(rbind(paste0(labels_block, "_OL"), paste0(labels_block, "_HS")))
    rows <- apply(wide, 1, function(r) {
      vals <- ifelse(is.na(r[cols]), "--", r[cols])
      paste(c(as.character(r[["term_clean"]]), vals), collapse = " & ")
    })
    paste(paste0(rows, " \\\\"), collapse = "\n")
  })
}

################################################################################
### paper_table_to_latex()
###
### Renders a single-model-style paper table (term_clean, OL/HS
### estimate_fmt + se_fmt) as LaTeX rows, one row per term.
################################################################################

paper_table_to_latex <- function(paper_table) {

  library(tidyverse)

  latex_rows <- paper_table %>%
    mutate(
      latex_row = paste0(
        term_clean, " & ",
        OL_estimate_fmt, " & ", OL_se_fmt, " & ",
        HS_estimate_fmt, " & ", HS_se_fmt, " \\\\"
      )
    ) %>%
    pull(latex_row)

  paste(latex_rows, collapse = "\n")
}

################################################################################
### wide_table_to_standalone_tex()
###
### Wraps wide_table_to_latex()'s rows in a compilable standalone .tex
### document, so the table can be checked with pdflatex on its own.
################################################################################

wide_table_to_standalone_tex <- function(wide, model_labels, models_per_block = length(model_labels)) {

  library(tidyverse)

  body_blocks <- wide_table_to_latex(wide, model_labels, models_per_block)

  header1 <- paste(
    c("Term", map_chr(model_labels, ~ paste0("\\multicolumn{2}{c}{", .x, "}"))),
    collapse = " & "
  )
  header2 <- paste(c("", rep(c("OL", "HS"), length(model_labels))), collapse = " & ")
  colspec <- paste0("l", strrep("cc", length(model_labels)))

  ### Flag any model whose Observations fall short of the modal count
  ### (e.g. a missing response), so it doesn't have to be caught by hand.
  obs_row <- wide %>% filter(term_clean == "Observations")
  obs_by_model <- setNames(
    as.integer(gsub(",", "", trimws(unlist(obs_row[paste0(model_labels, "_OL")])))),
    model_labels
  )
  typical_n <- as.integer(names(sort(table(obs_by_model), decreasing = TRUE))[1])
  short_models <- names(obs_by_model)[obs_by_model != typical_n]

  note <- character(0)
  if (length(short_models) > 0) {
    detail <- paste0(
      short_models, " (", format(obs_by_model[short_models], big.mark = ","),
      ", ", typical_n - obs_by_model[short_models], " fewer)"
    )
    total_missing <- sum(typical_n - obs_by_model[short_models])
    missing_phrase <- if (total_missing == 1) "a missing response" else "missing responses"
    note <- c(
      "\\smallskip",
      "\\begin{minipage}{\\textwidth}",
      "\\raggedright",
      paste0(
        "\\footnotesize \\textit{Note}: Observations = ", format(typical_n, big.mark = ","),
        " per model, except ", paste(detail, collapse = "; "),
        ", due to ", missing_phrase, "."
      ),
      "\\end{minipage}"
    )
  }

  c(
    "\\documentclass{article}",
    "\\usepackage[margin=0.5in]{geometry}",
    "\\usepackage{booktabs}",
    "\\usepackage{amsmath}",
    "\\usepackage{graphicx}",
    "\\begin{document}",
    "\\begin{table}",
    "\\centering",
    "\\resizebox{\\textwidth}{!}{%",
    paste0("\\begin{tabular}{", colspec, "}"),
    "\\toprule",
    paste0(header1, " \\\\"),
    paste0(header2, " \\\\"),
    "\\midrule",
    body_blocks,
    "\\bottomrule",
    "\\end{tabular}%",
    "}",
    note,
    "\\end{table}",
    "\\end{document}"
  )
}

################################################################################
### paper_table_to_standalone_tex()
###
### Wraps paper_table_to_latex()'s rows in a compilable standalone .tex doc.
################################################################################

paper_table_to_standalone_tex <- function(paper_table) {

  body_rows <- paper_table_to_latex(paper_table)

  c(
    "\\documentclass{article}",
    "\\usepackage{booktabs}",
    "\\usepackage{amsmath}",
    "\\begin{document}",
    "\\begin{table}",
    "\\centering",
    "\\begin{tabular}{lcccc}",
    "\\toprule",
    "Term & OL Estimate & OL SE & HS Estimate & HS SE \\\\",
    "\\midrule",
    body_rows,
    "\\bottomrule",
    "\\end{tabular}",
    "\\end{table}",
    "\\end{document}"
  )
}
