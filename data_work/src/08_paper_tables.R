################################################################################
### 08_paper_tables.R  --  remaining descriptive tables in the manuscript
###
### Produces Table 1 (evaluation framework) from outputs of stages 01, 03, 04,
### and 06, plus the annotation-cost appendix table from documented token counts,
### endpoint prices, and three independent runs.
################################################################################

BASEDIR <- "/Users/sseckman/Downloads/Annotation_Sensitivity_in_Large_Language_Models__ACL_ (1)"
source(file.path(BASEDIR, "data_work", "src", "00_utils.R"))

say <- make_logger("08_paper_tables.log")

need <- function(filename) {
  p <- file.path(OUT, filename)
  if (!file.exists(p)) stop("Required upstream output is missing: ", p)
  readr::read_csv(p, show_col_types = FALSE)
}

### ------------------------------------------------------- framework summary
within <- need("ladder_fleiss_within.csv")
across <- need("ladder_cohen_across.csv")
human <- need("ladder_cohen_human.csv")
prev <- need("deff_run_prevalence.csv")
decomp <- need("deff_components.csv")
human_comp <- need("human_llm_components.csv")

fmt2 <- function(x) sub("^0", "", sprintf("%.2f", x))
gpt_range <- function(x) paste0(fmt2(min(x)), "--", fmt2(max(x)))

within_med <- median(within$kappa, na.rm = TRUE)
across_med <- median(across$kappa, na.rm = TRUE)
human_med <- median(human$kappa, na.rm = TRUE)
gpt_within <- gpt_range(within$kappa[within$model == "GPT-4o-mini"])
gpt_human <- gpt_range(human$kappa[human$model == "GPT-4o-mini"])

hs_model <- prev %>% filter(outcome == "HS") %>% group_by(model) %>%
  summarise(prevalence = mean(prev), .groups = "drop")
hs_range <- sprintf("%d\\%%--%d\\%%", round(100 * min(hs_model$prevalence)),
                    round(100 * max(hs_model$prevalence)))

llm_final <- decomp %>% filter(component == "model") %>% pull(cumulative_deff)
human_deff <- human_comp$deff_h
llm_range <- sprintf("%d--%d", min(round(llm_final)), max(round(llm_final)))
human_range <- sprintf("%d--%d", min(round(human_deff)), max(round(human_deff)))

ROW_END <- intToUtf8(c(92L, 92L))
framework_lines <- c(
  paste0(r"(Validity & Outputs vs. intended construct & Indirectly constrained by high sensitivity. )", ROW_END),
  paste0(sprintf(r"(Performance & Model vs. human benchmark & Cohen's $\kappa$ = %s median (GPT-4o-mini %s). )",
                 fmt2(human_med), gpt_human), ROW_END),
  paste0(sprintf(r"(Reliability & Same model/design, repeated runs & Fleiss' $\kappa$ = %s median (GPT-4o-mini %s). )",
                 fmt2(within_med), gpt_within), ROW_END),
  paste0(sprintf(r"(Sensitivity & Across task designs \& models & Cohen's $\kappa$ = %s median across designs; HS rates %s. )",
                 fmt2(across_med), hs_range), ROW_END),
  paste0(sprintf(r"(\textit{Human reference} & \textit{Across survey versions} & Design effect %s on same items, vs. %s for LLMs. )",
                 human_range, llm_range), ROW_END))
write_tabular(framework_lines, "table_framework.tex",
              "@{}p{1.3cm}p{2.0cm}p{3.45cm}@{}", header =
                paste0(r"(\textbf{Criterion} & \textbf{Comparison} & \textbf{Our evidence} )", ROW_END))

say("Framework: reliability %s, sensitivity %s, performance %s; deff humans %s, LLMs %s",
    fmt2(within_med), fmt2(across_med), fmt2(human_med), human_range, llm_range)

### ---------------------------------------------------------- annotation costs
## Token counts are the rounded counts reported in the appendix. Prices are USD
## per million tokens at collection: GPT-4o-mini $0.15 input/$0.60 output;
## GPT-5.4 $2.50 input/$15 output. Costs cover all three independent runs even
## though API calls are displayed per run, matching the experiment's total cost.
cost_spec <- tibble::tribble(
  ~condition,           ~calls, ~task_tokens, ~tweet_tokens, ~output_tokens,
  "Indiv., no conf.",      3000,          517,            19,              2,
  "Indiv., w/ conf.",      3000,          580,            19,              4,
  "Batch, no conf.",        500,          639,           112,             12,
  "Batch, w/ conf.",        500,          724,           112,             24)
prices <- tibble::tribble(
  ~model,        ~input_per_million, ~output_per_million,
  "GPT-4o-mini",               0.15,                 0.60,
  "GPT-5.4",                   2.50,                15.00)
N_RUNS <- 3
cost_long <- tidyr::crossing(cost_spec, prices) %>%
  mutate(cost = N_RUNS * calls *
           ((task_tokens + tweet_tokens) * input_per_million +
              output_tokens * output_per_million) / 1e6)
cost_wide <- cost_long %>% select(condition, model, cost) %>%
  tidyr::pivot_wider(names_from = model, values_from = cost) %>%
  left_join(cost_spec, by = "condition") %>%
  mutate(order = match(condition, cost_spec$condition)) %>% arrange(order)

cost_lines <- vapply(seq_len(nrow(cost_wide)), function(i) paste0(sprintf(
  "%s & %d & %d & %d & %d & %.2f & %.2f ",
  cost_wide$condition[i], cost_wide$calls[i], cost_wide$task_tokens[i],
  cost_wide$tweet_tokens[i], cost_wide$output_tokens[i],
  cost_wide$`GPT-4o-mini`[i], cost_wide$`GPT-5.4`[i]), ROW_END), character(1))
write_tabular(cost_lines, "table_costs.tex", "lcccccc", header = c(
  paste0(r"( & \textbf{API} & \multicolumn{3}{c}{\textbf{Tokens per call}} & \multicolumn{2}{c}{\textbf{Cost (\$), 3 runs}} )", ROW_END),
  r"(\cmidrule(lr){3-5} \cmidrule(lr){6-7})",
  paste0(r"(\textbf{Condition} & \textbf{Calls/run} & \textbf{Task} & \textbf{Tweet} & \textbf{Out} & \textbf{4o-mini} & \textbf{5.4} )", ROW_END)))

readr::write_csv(cost_long, file.path(OUT, "annotation_costs.csv"))
say("Wrote table_framework.tex and table_costs.tex (cost inputs exported to annotation_costs.csv).")
say("Note: rounded token counts imply $0.18, not $0.19, for GPT-4o-mini batch/no-confidence.")
