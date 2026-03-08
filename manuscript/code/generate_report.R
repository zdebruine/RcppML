#!/usr/bin/env Rscript
# Generate plots and markdown report from benchmark results

library(ggplot2)
library(dplyr)
library(tidyr)

cat("=== Generating Benchmark Report ===\n")

# Load results
results <- read.csv("benchmark_final.csv", stringsAsFactors = FALSE)
summary_stats <- read.csv("benchmark_final_summary.csv", stringsAsFactors = FALSE)

cat("Loaded", nrow(results), "results\n")

# Theme
theme_set(theme_minimal(base_size = 11) + 
            theme(legend.position = "bottom",
                  strip.background = element_rect(fill = "grey90", color = NA),
                  panel.grid.minor = element_blank()))

# Color palettes
method_colors <- c("RcppML" = "#2E86AB", "singlet" = "#E94F37")
dataset_colors <- c("movielens" = "#1f77b4", "hawaiibirds" = "#ff7f0e", 
                    "aml" = "#2ca02c", "pbmc3k" = "#d62728",
                    "olivetti" = "#9467bd", "digits_full" = "#8c564b",
                    "golub" = "#e377c2")

# ========================================
# PLOT 1: Runtime comparison (Standard NMF)
# ========================================
cat("Creating runtime plots...\n")

std_runtime <- summary_stats %>%
  filter(variant == "standard", mask_zeros == FALSE)

if (nrow(std_runtime) > 0) {
  p1 <- std_runtime %>%
    ggplot(aes(x = factor(rank), y = mean_time * 1000, fill = method)) +
    geom_bar(stat = "identity", position = position_dodge(width = 0.8), width = 0.7) +
    geom_errorbar(aes(ymin = pmax(0, (mean_time - sd_time) * 1000),
                      ymax = (mean_time + sd_time) * 1000),
                  position = position_dodge(width = 0.8), width = 0.25) +
    facet_wrap(~dataset, scales = "free_y", ncol = 4) +
    scale_fill_manual(values = method_colors) +
    labs(title = "Standard NMF: Total Runtime by Dataset and Rank",
         subtitle = "5 iterations, triplicate runs",
         x = "Rank (k)", y = "Total Runtime (ms)", fill = "Method") +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  
  ggsave("plot_standard_runtime.png", p1, width = 14, height = 8, dpi = 150)
  cat("Saved plot_standard_runtime.png\n")
}

# ========================================
# PLOT 2: CV NMF Runtime comparison
# ========================================
cv_runtime <- summary_stats %>%
  filter(variant == "cv", mask_zeros == FALSE, precision == "double")

if (nrow(cv_runtime) > 0) {
  p2 <- cv_runtime %>%
    ggplot(aes(x = factor(rank), y = mean_time * 1000, fill = method)) +
    geom_bar(stat = "identity", position = position_dodge(width = 0.8), width = 0.7) +
    geom_errorbar(aes(ymin = pmax(0, (mean_time - sd_time) * 1000),
                      ymax = (mean_time + sd_time) * 1000),
                  position = position_dodge(width = 0.8), width = 0.25) +
    facet_wrap(~dataset, scales = "free_y", ncol = 4) +
    scale_fill_manual(values = method_colors) +
    labs(title = "CV NMF: Total Runtime by Dataset and Rank",
         subtitle = "5 iterations, test_fraction=1/16, triplicate runs",
         x = "Rank (k)", y = "Total Runtime (ms)", fill = "Method") +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  
  ggsave("plot_cv_runtime.png", p2, width = 14, height = 8, dpi = 150)
  cat("Saved plot_cv_runtime.png\n")
}

# ========================================
# PLOT 3: Speedup (singlet / RcppML)
# ========================================
speedup_data <- results %>%
  filter(variant == "standard", mask_zeros == FALSE) %>%
  select(dataset, rank, replicate, method, total_time) %>%
  pivot_wider(names_from = method, values_from = total_time) %>%
  filter(!is.na(singlet) & !is.na(RcppML) & RcppML > 0) %>%
  mutate(speedup = singlet / RcppML)

if (nrow(speedup_data) > 0) {
  speedup_summary <- speedup_data %>%
    group_by(dataset, rank) %>%
    summarise(
      mean_speedup = mean(speedup, na.rm = TRUE),
      sd_speedup = sd(speedup, na.rm = TRUE),
      .groups = "drop"
    )
  
  p3 <- speedup_summary %>%
    ggplot(aes(x = factor(rank), y = mean_speedup, fill = dataset)) +
    geom_bar(stat = "identity", position = position_dodge(width = 0.8), width = 0.7) +
    geom_errorbar(aes(ymin = mean_speedup - sd_speedup, ymax = mean_speedup + sd_speedup),
                  position = position_dodge(width = 0.8), width = 0.25) +
    geom_hline(yintercept = 1, linetype = "dashed", color = "red") +
    scale_fill_manual(values = dataset_colors) +
    labs(title = "RcppML Speedup over singlet (Standard NMF)",
         subtitle = "Values > 1 indicate RcppML is faster",
         x = "Rank (k)", y = "Speedup (singlet time / RcppML time)", fill = "Dataset") +
    theme(legend.position = "right")
  
  ggsave("plot_speedup.png", p3, width = 12, height = 6, dpi = 150)
  cat("Saved plot_speedup.png\n")
}

# ========================================
# PLOT 4: sparse mode comparison (RcppML only)
# ========================================
sparse_comparison <- summary_stats %>%
  filter(variant == "standard", method == "RcppML") %>%
  mutate(sparse_mode = ifelse(mask_zeros, "sparse=TRUE", "sparse=FALSE"))

if (nrow(sparse_comparison) > 0) {
  p4 <- sparse_comparison %>%
    ggplot(aes(x = factor(rank), y = mean_time * 1000, fill = sparse_mode)) +
    geom_bar(stat = "identity", position = position_dodge(width = 0.8), width = 0.7) +
    geom_errorbar(aes(ymin = pmax(0, (mean_time - sd_time) * 1000),
                      ymax = (mean_time + sd_time) * 1000),
                  position = position_dodge(width = 0.8), width = 0.25) +
    facet_wrap(~dataset, scales = "free_y", ncol = 4) +
    scale_fill_brewer(palette = "Set2") +
    labs(title = "RcppML Standard NMF: sparse Mode Effect",
         x = "Rank (k)", y = "Total Runtime (ms)", fill = "Setting") +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  
  ggsave("plot_sparse_effect.png", p4, width = 14, height = 8, dpi = 150)
  cat("Saved plot_sparse_effect.png\n")
}

# ========================================
# PLOT 5: Float vs Double precision (RcppML CV)
# ========================================
precision_data <- summary_stats %>%
  filter(variant == "cv", mask_zeros == FALSE, method == "RcppML")

if (nrow(precision_data) > 0) {
  p5 <- precision_data %>%
    ggplot(aes(x = factor(rank), y = mean_time * 1000, fill = precision)) +
    geom_bar(stat = "identity", position = position_dodge(width = 0.8), width = 0.7) +
    geom_errorbar(aes(ymin = pmax(0, (mean_time - sd_time) * 1000),
                      ymax = (mean_time + sd_time) * 1000),
                  position = position_dodge(width = 0.8), width = 0.25) +
    facet_wrap(~dataset, scales = "free_y", ncol = 4) +
    scale_fill_brewer(palette = "Set1") +
    labs(title = "RcppML CV NMF: Float vs Double Precision Runtime",
         x = "Rank (k)", y = "Total Runtime (ms)", fill = "Precision") +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  
  ggsave("plot_precision_runtime.png", p5, width = 14, height = 8, dpi = 150)
  cat("Saved plot_precision_runtime.png\n")
}

# ========================================
# PLOT 6: Test MSE comparison (CV)
# ========================================
test_mse_data <- results %>%
  filter(cv == TRUE, precision == "double", mask_zeros == FALSE, !is.na(test_mse)) %>%
  select(dataset, rank, replicate, method, test_mse)

if (nrow(test_mse_data) > 0) {
  p6 <- test_mse_data %>%
    ggplot(aes(x = factor(rank), y = test_mse, fill = method)) +
    geom_boxplot(position = position_dodge(width = 0.8), width = 0.6) +
    facet_wrap(~dataset, scales = "free_y", ncol = 4) +
    scale_fill_manual(values = method_colors) +
    labs(title = "CV NMF: Test MSE by Dataset and Rank",
         x = "Rank (k)", y = "Test MSE", fill = "Method") +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  
  ggsave("plot_test_mse.png", p6, width = 14, height = 8, dpi = 150)
  cat("Saved plot_test_mse.png\n")
}

# ========================================
# PLOT 7: Train MSE comparison (Standard NMF)
# ========================================
train_mse_data <- results %>%
  filter(variant == "standard", mask_zeros == FALSE, !is.na(train_mse)) %>%
  select(dataset, rank, replicate, method, train_mse)

if (nrow(train_mse_data) > 0) {
  p7 <- train_mse_data %>%
    ggplot(aes(x = factor(rank), y = train_mse, fill = method)) +
    geom_boxplot(position = position_dodge(width = 0.8), width = 0.6) +
    facet_wrap(~dataset, scales = "free_y", ncol = 4) +
    scale_fill_manual(values = method_colors) +
    labs(title = "Standard NMF: Train MSE by Dataset and Rank",
         x = "Rank (k)", y = "Train MSE", fill = "Method") +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  
  ggsave("plot_train_mse.png", p7, width = 14, height = 8, dpi = 150)
  cat("Saved plot_train_mse.png\n")
}

# ========================================
# PLOT 8: Aggregate speedup bar chart
# ========================================
agg_speedup <- speedup_data %>%
  group_by(rank) %>%
  summarise(
    mean_speedup = mean(speedup, na.rm = TRUE),
    se_speedup = sd(speedup, na.rm = TRUE) / sqrt(n()),
    .groups = "drop"
  )

if (nrow(agg_speedup) > 0) {
  p8 <- agg_speedup %>%
    ggplot(aes(x = factor(rank), y = mean_speedup)) +
    geom_bar(stat = "identity", fill = "#2E86AB", width = 0.7) +
    geom_errorbar(aes(ymin = mean_speedup - se_speedup, ymax = mean_speedup + se_speedup),
                  width = 0.25) +
    geom_hline(yintercept = 1, linetype = "dashed", color = "red") +
    labs(title = "RcppML Aggregate Speedup over singlet",
         subtitle = "Standard NMF, all datasets combined",
         x = "Rank (k)", y = "Speedup Factor")
  
  ggsave("plot_aggregate_speedup.png", p8, width = 8, height = 5, dpi = 150)
  cat("Saved plot_aggregate_speedup.png\n")
}

cat("\nAll plots generated.\n")

# ========================================
# Generate Markdown Report
# ========================================
cat("\nGenerating Markdown report...\n")

# Dataset info
datasets_info <- results %>%
  group_by(dataset) %>%
  summarise(n = n()) %>%
  arrange(desc(n))

# Overall summary stats
overall_std <- results %>%
  filter(variant == "standard", mask_zeros == FALSE) %>%
  group_by(method) %>%
  summarise(
    total_runs = n(),
    mean_time_ms = mean(total_time * 1000, na.rm = TRUE),
    median_time_ms = median(total_time * 1000, na.rm = TRUE),
    mean_train_mse = mean(train_mse, na.rm = TRUE),
    .groups = "drop"
  )

overall_cv <- results %>%
  filter(variant == "cv", mask_zeros == FALSE, precision == "double") %>%
  group_by(method) %>%
  summarise(
    total_runs = n(),
    mean_time_ms = mean(total_time * 1000, na.rm = TRUE),
    mean_test_mse = mean(test_mse, na.rm = TRUE),
    .groups = "drop"
  )

# Calculate overall speedup
rcppml_std_time <- overall_std$mean_time_ms[overall_std$method == "RcppML"]
singlet_std_time <- overall_std$mean_time_ms[overall_std$method == "singlet"]
overall_speedup <- singlet_std_time / rcppml_std_time

md_lines <- c(
  "# Comprehensive Benchmark: singlet vs RcppML",
  "",
  paste0("**Generated:** ", Sys.time()),
  "",
  "## Overview",
  "",
  "This benchmark compares the NMF implementations in singlet and RcppML across multiple datasets and configurations.",
  "",
  "### Configuration",
  "",
  "- **Datasets:** movielens, hawaiibirds, aml, pbmc3k, olivetti, digits_full, golub",
  "- **Ranks (k):** 2, 4, 8, 16, 32, 64",
  "- **Iterations:** 5 per run",
  "- **Replicates:** 3 per configuration",
  "- **CV test fraction:** 1/16 (~6.25%)",
  "- **Tolerance:** 1e-10",
  "",
  "### Test Variants",
  "",
  "1. **Standard NMF (sparse=FALSE)** - RcppML & singlet",
  "2. **Standard NMF (sparse=TRUE)** - RcppML only",
  "3. **CV NMF (double precision)** - RcppML & singlet",
  "4. **CV NMF (sparse=TRUE)** - RcppML only",
  "5. **CV NMF (float precision)** - RcppML only",
  "",
  "---",
  "",
  "## Key Findings",
  "",
  sprintf("### Overall Performance (Standard NMF)"),
  "",
  sprintf("| Metric | RcppML | singlet |"),
  sprintf("|--------|--------|---------|"),
  sprintf("| Mean Runtime (ms) | %.2f | %.2f |", rcppml_std_time, singlet_std_time),
  sprintf("| **Overall Speedup** | **%.2fx** | baseline |", overall_speedup),
  ""
)

# Per-rank speedup table
if (nrow(agg_speedup) > 0) {
  md_lines <- c(md_lines,
    "### Speedup by Rank",
    "",
    "| Rank | Mean Speedup | SE |",
    "|------|--------------|-----|"
  )
  
  for (i in 1:nrow(agg_speedup)) {
    md_lines <- c(md_lines,
      sprintf("| %d | %.2fx | ±%.2f |", 
              agg_speedup$rank[i], 
              agg_speedup$mean_speedup[i],
              agg_speedup$se_speedup[i]))
  }
  md_lines <- c(md_lines, "")
}

# CV NMF results
if (nrow(overall_cv) > 0) {
  rcppml_cv <- overall_cv %>% filter(method == "RcppML")
  singlet_cv <- overall_cv %>% filter(method == "singlet")
  
  if (nrow(rcppml_cv) > 0 && nrow(singlet_cv) > 0) {
    cv_speedup <- singlet_cv$mean_time_ms / rcppml_cv$mean_time_ms
    md_lines <- c(md_lines,
      "### CV NMF Performance",
      "",
      sprintf("| Metric | RcppML | singlet |"),
      sprintf("|--------|--------|---------|"),
      sprintf("| Mean Runtime (ms) | %.2f | %.2f |", rcppml_cv$mean_time_ms, singlet_cv$mean_time_ms),
      sprintf("| Mean Test MSE | %.4f | %.4f |", rcppml_cv$mean_test_mse, singlet_cv$mean_test_mse),
      sprintf("| **CV Speedup** | **%.2fx** | baseline |", cv_speedup),
      ""
    )
  }
}

md_lines <- c(md_lines,
  "---",
  "",
  "## Detailed Results",
  "",
  "### Standard NMF Runtime (ms)",
  ""
)

# Standard NMF table
std_table <- summary_stats %>%
  filter(variant == "standard", mask_zeros == FALSE) %>%
  select(dataset, rank, method, mean_time, sd_time) %>%
  mutate(time_str = sprintf("%.1f ± %.1f", mean_time * 1000, sd_time * 1000)) %>%
  select(dataset, rank, method, time_str) %>%
  pivot_wider(names_from = method, values_from = time_str)

md_lines <- c(md_lines,
  "| Dataset | Rank | RcppML | singlet |",
  "|---------|------|--------|---------|"
)

for (i in 1:nrow(std_table)) {
  md_lines <- c(md_lines,
    sprintf("| %s | %d | %s | %s |", 
            std_table$dataset[i], 
            std_table$rank[i],
            ifelse(is.na(std_table$RcppML[i]), "N/A", std_table$RcppML[i]),
            ifelse(is.na(std_table$singlet[i]), "N/A", std_table$singlet[i])))
}

# Float vs Double precision section
precision_summary <- summary_stats %>%
  filter(variant == "cv", mask_zeros == FALSE, method == "RcppML") %>%
  select(dataset, rank, precision, mean_time, mean_test_mse) %>%
  mutate(
    time_str = sprintf("%.1f", mean_time * 1000),
    mse_str = sprintf("%.4f", mean_test_mse)
  )

double_data <- precision_summary %>% 
  filter(precision == "double") %>%
  select(dataset, rank, double_time = time_str, double_mse = mse_str)

float_data <- precision_summary %>%
  filter(precision == "float") %>%
  select(dataset, rank, float_time = time_str, float_mse = mse_str)

precision_table <- left_join(double_data, float_data, by = c("dataset", "rank"))

md_lines <- c(md_lines,
  "",
  "### Float vs Double Precision (RcppML CV NMF)",
  "",
  "| Dataset | Rank | Double Time (ms) | Float Time (ms) | Double Test MSE | Float Test MSE |",
  "|---------|------|------------------|-----------------|-----------------|----------------|"
)

for (i in 1:nrow(precision_table)) {
  md_lines <- c(md_lines,
    sprintf("| %s | %d | %s | %s | %s | %s |",
            precision_table$dataset[i],
            precision_table$rank[i],
            ifelse(is.na(precision_table$double_time[i]), "N/A", precision_table$double_time[i]),
            ifelse(is.na(precision_table$float_time[i]), "N/A", precision_table$float_time[i]),
            ifelse(is.na(precision_table$double_mse[i]), "N/A", precision_table$double_mse[i]),
            ifelse(is.na(precision_table$float_mse[i]), "N/A", precision_table$float_mse[i])))
}

md_lines <- c(md_lines,
  "",
  "---",
  "",
  "## Plots",
  "",
  "### Runtime Comparisons",
  "",
  "#### Standard NMF Runtime",
  "![Standard NMF Runtime](plot_standard_runtime.png)",
  "",
  "#### CV NMF Runtime", 
  "![CV NMF Runtime](plot_cv_runtime.png)",
  "",
  "#### RcppML Speedup over singlet",
  "![Speedup](plot_speedup.png)",
  "",
  "#### Aggregate Speedup by Rank",
  "![Aggregate Speedup](plot_aggregate_speedup.png)",
  "",
  "### MSE Comparisons",
  "",
  "#### Train MSE (Standard NMF)",
  "![Train MSE](plot_train_mse.png)",
  "",
  "#### Test MSE (CV NMF)",
  "![Test MSE](plot_test_mse.png)",
  "",
  "### RcppML-Specific Comparisons",
  "",
  "#### sparse Mode Effect",
  "![sparse Effect](plot_sparse_effect.png)",
  "",
  "#### Float vs Double Precision",
  "![Precision Runtime](plot_precision_runtime.png)",
  "",
  "---",
  "",
  "*Report generated by generate_report.R*"
)

writeLines(md_lines, "BENCHMARK_RESULTS.md")
cat("Saved BENCHMARK_RESULTS.md\n")

cat("\n=== Report Generation Complete ===\n")
