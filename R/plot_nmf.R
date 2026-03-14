# Suppress R CMD check NOTEs for ggplot2 aes() variables
utils::globalVariables(c("k", "type", "mean_loss"))

#' Plot NMF Training History and Diagnostics
#'
#' TensorBoard-style visualization of NMF training dynamics, convergence analysis,
#' and factor diagnostics. Provides multiple plot types for comprehensive model analysis.
#'
#' @param x object of class "nmf"
#' @param type plot type: 
#'   - "loss": Loss components over iterations (default)
#'   - "convergence": Log-scale loss convergence
#'   - "regularization": Regularization penalty contributions
#'   - "sparsity": Factor sparsity patterns
#' @param smooth apply smoothing (LOESS) for noisy curves (default TRUE)
#' @param span smoothing span for LOESS (default 0.3)
#' @param log_scale use log scale for y-axis (default FALSE, auto TRUE for "convergence")
#' @param interactive create interactive plotly plot (default FALSE)
#' @param theme ggplot2 theme: "classic", "minimal", "dark" (default "classic")
#' @param ... additional arguments passed to specific plotting functions
#' @return ggplot2 or plotly object
#' @seealso \code{\link{nmf}}, \code{\link{compare_nmf}}
#' @export
#' @method plot nmf
#' @examples
#' \donttest{
#' # Basic loss plot
#' model <- nmf(hawaiibirds, k = 10)
#' plot(model)
#'
#' # Convergence analysis
#' plot(model, type = "convergence")
#'
#' # Interactive plot
#' plot(model, type = "loss", interactive = TRUE)
#'
#' # Compare multiple runs
#' models <- replicate(5, nmf(hawaiibirds, k = 10), simplify = FALSE)
#' plot(models[[1]], type = "sparsity")
#' }
plot.nmf <- function(x, type = c("loss", "convergence", "regularization", "sparsity"),
                     smooth = TRUE, span = 0.3, log_scale = FALSE,
                     interactive = FALSE, theme = "classic", ...) {
  
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("ggplot2 package required for plotting. Install with: install.packages('ggplot2')")
  }
  
  type <- match.arg(type)
  
  # Dispatch to specific plot type
  if (type == "loss") {
    return(plot_nmf_loss(x, smooth, span, log_scale, interactive, theme))
  } else if (type == "convergence") {
    return(plot_nmf_convergence(x, smooth, span, interactive, theme))
  } else if (type == "regularization") {
    return(plot_nmf_regularization(x, smooth, span, interactive, theme))
  } else if (type == "sparsity") {
    return(plot_nmf_sparsity(x, interactive, theme))
  }
}

#' Plot Loss History
#' 
#' @return A \code{ggplot2} object (or \code{plotly} if \code{interactive = TRUE})
#' @keywords internal
plot_nmf_loss <- function(x, smooth, span, log_scale, interactive, theme) {
  # Check if loss history exists
  if (is.null(x@misc$loss_history)) {
    stop("No loss history found. Run nmf() with track_loss = TRUE")
  }
  
  # Extract loss history — stored as a numeric vector in misc$loss_history
  # with optional test loss in misc$test_loss_history
  train_loss <- x@misc$loss_history
  test_loss <- x@misc$test_loss_history
  n_iter <- length(train_loss)
  
  if (n_iter == 0) {
    stop("Loss history is empty")
  }
  
  # Build data frame in long format manually
  df_list <- list()
  
  # Add train loss
  if (!is.null(train_loss) && length(train_loss) > 0) {
    df_list[[length(df_list) + 1]] <- data.frame(
      iteration = seq_len(length(train_loss)),
      loss_type = "train_loss",
      loss = train_loss,
      stringsAsFactors = FALSE
    )
  }
  
  # Add test loss if exists
  if (!is.null(test_loss) && length(test_loss) > 0 && any(test_loss > 0)) {
    df_list[[length(df_list) + 1]] <- data.frame(
      iteration = seq_len(length(test_loss)),
      loss_type = "test_loss",
      loss = test_loss,
      stringsAsFactors = FALSE
    )
  }
  
  # Add total loss (currently not stored separately — skip)
  
  df_long <- do.call(rbind, df_list)
  
  # Apply smoothing if requested
  if (smooth && nrow(df_long) > 3) {
    df_long <- do.call(rbind, lapply(split(df_long, df_long$loss_type), function(sub_df) {
      if (nrow(sub_df) > 3) {
        tryCatch({
          smoothed <- stats::loess(loss ~ iteration, data = sub_df, span = span)
          sub_df$loss_smooth <- predict(smoothed)
        }, error = function(e) {
          sub_df$loss_smooth <- sub_df$loss
        })
      } else {
        sub_df$loss_smooth <- sub_df$loss
      }
      sub_df
    }))
  }
  
  # Create ggplot
  p <- ggplot2::ggplot(df_long, ggplot2::aes(x = iteration, y = loss, color = loss_type))
  
  if (smooth) {
    p <- p + 
      ggplot2::geom_line(ggplot2::aes(y = loss), alpha = 0.3, linewidth = 0.5) +
      ggplot2::geom_line(ggplot2::aes(y = loss_smooth), linewidth = 1)
  } else {
    p <- p + ggplot2::geom_line(linewidth = 1)
  }
  
  p <- p +
    ggplot2::labs(
      title = "NMF Training History",
      subtitle = paste0("Loss function: ", x@misc$loss_type, 
                       ", Rank: ", ncol(x@w),
                       ", Final loss: ", format(x@misc$loss, digits = 4)),
      x = "Iteration",
      y = if (log_scale) "Loss (log scale)" else "Loss",
      color = "Loss Type"
    ) +
    ggplot2::scale_color_manual(
      values = c("train_loss" = "#1f77b4", "test_loss" = "#ff7f0e", 
                 "total_loss" = "#2ca02c"),
      labels = c("train_loss" = "Training", "test_loss" = "Validation", 
                 "total_loss" = "Total (with penalties)")
    )
  
  if (log_scale) {
    p <- p + ggplot2::scale_y_log10()
  }
  
  # Apply theme
  p <- apply_plot_theme(p, theme)
  
  # Convert to plotly if interactive
  if (interactive && requireNamespace("plotly", quietly = TRUE)) {
    p <- plotly::ggplotly(p)
  }
  
  return(p)
}

#' Plot Convergence Analysis
#' 
#' @return A \code{ggplot2} object (or \code{plotly} if \code{interactive = TRUE})
#' @keywords internal
plot_nmf_convergence <- function(x, smooth, span, interactive, theme) {
  if (is.null(x@misc$loss_history)) {
    stop("No loss history found. Run nmf() with track_loss = TRUE")
  }
  
  history <- x@misc$loss_history
  n_iter <- length(history)
  
  # Compute relative change between iterations
  train_loss <- history
  rel_change <- c(NA, abs(diff(train_loss)) / (train_loss[-length(train_loss)] + 1e-10))
  
  df <- data.frame(
    iteration = seq_len(n_iter),
    train_loss = train_loss,
    rel_change = rel_change
  )
  
  # Two-panel plot if patchwork available
  if (requireNamespace("patchwork", quietly = TRUE)) {
    p1 <- ggplot2::ggplot(df, ggplot2::aes(x = iteration, y = train_loss)) +
      ggplot2::geom_line(color = "#1f77b4", linewidth = 1) +
      ggplot2::scale_y_log10() +
      ggplot2::labs(x = "Iteration", y = "Training Loss (log scale)") +
      ggplot2::geom_vline(xintercept = x@misc$iter, linetype = "dashed", 
                         color = "red", alpha = 0.5) +
      ggplot2::annotate("text", x = x@misc$iter, y = max(train_loss),
                       label = paste("Converged at iter", x@misc$iter),
                       hjust = 1.1, color = "red")
    
    p2 <- ggplot2::ggplot(df[!is.na(df$rel_change), ], 
                          ggplot2::aes(x = iteration, y = rel_change)) +
      ggplot2::geom_line(color = "#ff7f0e", linewidth = 1) +
      ggplot2::scale_y_log10() +
      ggplot2::geom_hline(yintercept = x@misc$tol, linetype = "dashed", 
                         color = "red", alpha = 0.5) +
      ggplot2::annotate("text", x = max(df$iteration) * 0.8, y = x@misc$tol * 1.5,
                       label = paste0("tol = ", format(x@misc$tol, digits = 2)),
                       color = "red") +
      ggplot2::labs(x = "Iteration", y = "Relative Change (log scale)")
    
    p1 <- apply_plot_theme(p1, theme)
    p2 <- apply_plot_theme(p2, theme)
    
    p <- p1 / p2 + 
      patchwork::plot_annotation(
        title = "NMF Convergence Analysis",
        subtitle = paste0("Converged at iteration ", x@misc$iter, 
                         " with rel_change = ", format(x@misc$tol, digits = 4))
      )
  } else {
    # Single panel if patchwork not available
    p <- ggplot2::ggplot(df, ggplot2::aes(x = iteration, y = train_loss)) +
      ggplot2::geom_line(color = "#1f77b4", linewidth = 1) +
      ggplot2::scale_y_log10() +
      ggplot2::labs(
        title = "NMF Convergence",
        x = "Iteration",
        y = "Training Loss (log scale)"
      ) +
      ggplot2::geom_vline(xintercept = x@misc$iter, linetype = "dashed", color = "red")
    
    p <- apply_plot_theme(p, theme)
  }
  
  if (interactive && requireNamespace("plotly", quietly = TRUE)) {
    p <- plotly::ggplotly(p)
  }
  
  return(p)
}

#' Plot Regularization Breakdown
#' 
#' @return A \code{ggplot2} object showing regularization contributions
#' @keywords internal
plot_nmf_regularization <- function(x, smooth, span, interactive, theme) {
  if (is.null(x@misc$loss_history)) {
    stop("No loss history found. Run nmf() with track_loss = TRUE")
  }
  
  history <- x@misc$loss_history
  n_iter <- length(history)
  
  # Per-iteration penalty breakdown is not stored; show total loss curve
  # with annotation of which penalties were configured
  df <- data.frame(
    iteration = seq_len(n_iter),
    train_loss = history
  )
  
  # Detect active penalties from model parameters
  active_penalties <- character(0)
  if (!is.null(x@misc$L1) && any(x@misc$L1 > 0)) active_penalties <- c(active_penalties, "L1 (sparsity)")
  if (!is.null(x@misc$L2) && any(x@misc$L2 > 0)) active_penalties <- c(active_penalties, "L2 (ridge)")
  
  subtitle <- if (length(active_penalties) > 0) {
    paste("Active penalties:", paste(active_penalties, collapse = ", "))
  } else {
    "No regularization penalties active"
  }
  
  p <- ggplot2::ggplot(df, ggplot2::aes(x = iteration, y = train_loss)) +
    ggplot2::geom_line(linewidth = 1, color = "steelblue") +
    ggplot2::labs(
      title = "NMF Objective (with Regularization)",
      subtitle = subtitle,
      x = "Iteration",
      y = "Training Loss"
    )
  
  p <- apply_plot_theme(p, theme)
  
  if (interactive && requireNamespace("plotly", quietly = TRUE)) {
    p <- plotly::ggplotly(p)
  }
  
  return(p)
}

#' Plot Factor Sparsity
#'
#' @return A \code{ggplot2} object showing sparsity of W and H matrices
#' @keywords internal
plot_nmf_sparsity<- function(x, interactive, theme) {
  # Compute sparsity metrics for final W and H
  w_sparsity <- apply(x@w, 2, function(col) sum(col < 1e-10) / length(col))
  h_sparsity <- apply(x@h, 1, function(row) sum(row < 1e-10) / length(row))
  
  df <- data.frame(
    factor = rep(1:length(w_sparsity), 2),
    sparsity = c(w_sparsity, h_sparsity),
    matrix = rep(c("W", "H"), each = length(w_sparsity))
  )
  
  p <- ggplot2::ggplot(df, ggplot2::aes(x = factor, y = sparsity, fill = matrix)) +
    ggplot2::geom_col(position = "dodge") +
    ggplot2::labs(
      title = "NMF Factor Sparsity",
      subtitle = paste0("Rank: ", ncol(x@w)),
      x = "Factor",
      y = "Sparsity (fraction of zeros)",
      fill = "Matrix"
    ) +
    ggplot2::scale_fill_manual(values = c("W" = "#1f77b4", "H" = "#ff7f0e")) +
    ggplot2::scale_y_continuous(limits = c(0, 1),
      labels = if (requireNamespace("scales", quietly = TRUE)) scales::percent else ggplot2::waiver())
  
  p <- apply_plot_theme(p, theme)
  
  if (interactive && requireNamespace("plotly", quietly = TRUE)) {
    p <- plotly::ggplotly(p)
  }
  
  return(p)
}

#' Apply Plot Theme
#' 
#' @return A \code{ggplot2} theme object
#' @keywords internal
apply_plot_theme <- function(p, theme) {
  if (theme == "classic") {
    p <- p + ggplot2::theme_classic()
  } else if (theme == "minimal") {
    p <- p + ggplot2::theme_minimal()
  } else if (theme == "dark") {
    p <- p + ggplot2::theme_dark()
  }
  
  p <- p + ggplot2::theme(
    plot.title = ggplot2::element_text(face = "bold", size = 14),
    plot.subtitle = ggplot2::element_text(size = 10, color = "gray40"),
    legend.position = "bottom"
  )
  
  return(p)
}

#' Compare Multiple NMF Models
#'
#' Overlay training curves from multiple NMF models to compare convergence behavior.
#'
#' @param ... nmf objects to compare
#' @param labels optional character vector of model labels
#' @param metric what to compare: "loss" (default), "sparsity"
#' @param smooth apply smoothing (default TRUE)
#' @param span smoothing span for LOESS (default 0.3)
#' @return A \code{ggplot2} object comparing the requested metric across models.
#' @seealso \code{\link{nmf}}, \code{\link{plot.nmf}}
#' @export
#' @examples
#' \donttest{
#' library(Matrix)
#' A <- abs(rsparsematrix(100, 50, 0.2))
#' model1 <- nmf(A, k = 5, L1 = 0.01)
#' model2 <- nmf(A, k = 5, L1 = 0.1)
#' compare_nmf(model1, model2, labels = c("L1=0.01", "L1=0.1"))
#' }
compare_nmf <- function(..., labels = NULL, metric = "loss", smooth = TRUE, span = 0.3) {
  models <- list(...)
  
  if (length(models) == 0) {
    stop("No models provided")
  }
  
  if (is.null(labels)) {
    labels <- paste0("Model ", 1:length(models))
  }
  
  # Combine loss histories
  df_list <- lapply(seq_along(models), function(i) {
    m <- models[[i]]
    if (is.null(m@misc$loss_history)) {
      stop(sprintf("Model %d has no loss history. Use track_loss = TRUE", i))
    }
    history <- m@misc$loss_history
    test_loss <- m@misc$test_loss_history
    n_iter <- length(history)
    data.frame(
      model = labels[i],
      iteration = seq_len(n_iter),
      train_loss = history,
      test_loss = if (!is.null(test_loss) && length(test_loss) > 0) 
                    test_loss else rep(NA, n_iter),
      stringsAsFactors = FALSE
    )
  })
  
  df <- do.call(rbind, df_list)
  
  if (metric == "loss") {
    p <- ggplot2::ggplot(df, ggplot2::aes(x = iteration, y = train_loss, 
                                          color = model, group = model)) +
      ggplot2::geom_line(linewidth = 1) +
      ggplot2::labs(
        title = "NMF Model Comparison",
        x = "Iteration",
        y = "Training Loss",
        color = "Model"
      ) +
      ggplot2::theme_classic() +
      ggplot2::theme(legend.position = "bottom")
  } else if (metric == "sparsity") {
    # Compute per-model sparsity
    sp_list <- lapply(seq_along(models), function(i) {
      m <- models[[i]]
      w_sp <- mean(m@w < 1e-10)
      h_sp <- mean(m@h < 1e-10)
      data.frame(model = labels[i], matrix = c("W", "H"),
                 sparsity = c(w_sp, h_sp), stringsAsFactors = FALSE)
    })
    sp_df <- do.call(rbind, sp_list)
    p <- ggplot2::ggplot(sp_df, ggplot2::aes(x = model, y = sparsity, fill = matrix)) +
      ggplot2::geom_col(position = "dodge") +
      ggplot2::labs(title = "NMF Factor Sparsity Comparison",
                    x = "Model", y = "Sparsity (fraction zeros)") +
      ggplot2::theme_classic() +
      ggplot2::theme(legend.position = "bottom")
  } else {
    stop(sprintf("Unknown metric '%s'. Use 'loss' or 'sparsity'.", metric))
  }
  
  return(p)
}

#' Plot Cross-Validation Results
#'
#' Visualize NMF cross-validation results showing test (and optionally train)
#' loss across candidate ranks. Useful for selecting the optimal factorization rank.
#'
#' @param x object of class \code{nmfCrossValidate} (a data.frame from
#'   \code{nmf(k = 2:10, test_fraction = 0.1)})
#' @param show_train logical, overlay train loss on the plot (default TRUE if
#'   \code{train_mse} column is available and non-NA)
#' @param point_size size of data points (default 3)
#' @param interactive create interactive plotly plot (default FALSE)
#' @param ... additional arguments (unused)
#' @return ggplot2 or plotly object
#' @seealso \code{\link{nmf}}
#' @export
#' @method plot nmfCrossValidate
#' @examples
#' \donttest{
#' library(Matrix)
#' A <- abs(rsparsematrix(50, 30, 0.3))
#' cv_result <- nmf(A, k = 2:6, test_fraction = 0.1, cv_seed = 1, maxit = 20)
#' plot(cv_result)
#' }
plot.nmfCrossValidate <- function(x, show_train = NULL, point_size = 3,
                                   interactive = FALSE, ...) {
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("ggplot2 package required for plotting. Install with: install.packages('ggplot2')")
  }
  
  # Determine whether to show train loss
  has_train <- "train_mse" %in% names(x) && any(!is.na(x$train_mse))
  if (is.null(show_train)) show_train <- has_train
  
  # Build long-format data for plotting
  df_list <- list()
  
  # Test loss (always shown)
  df_list[[1]] <- data.frame(
    k = x$k,
    rep = x$rep,
    loss = x$test_mse,
    type = "Test",
    stringsAsFactors = FALSE
  )
  
  # Train loss (optional)
  if (show_train && has_train) {
    df_list[[2]] <- data.frame(
      k = x$k,
      rep = x$rep,
      loss = x$train_mse,
      type = "Train",
      stringsAsFactors = FALSE
    )
  }
  
  df <- do.call(rbind, df_list)
  
  # Compute mean per rank per type for the line
  agg_list <- list()
  for (type_val in unique(df$type)) {
    for (k_val in unique(df$k)) {
      subset_vals <- df$loss[df$type == type_val & df$k == k_val]
      agg_list[[length(agg_list) + 1]] <- data.frame(
        k = k_val,
        mean_loss = mean(subset_vals, na.rm = TRUE),
        type = type_val,
        stringsAsFactors = FALSE
      )
    }
  }
  df_mean <- do.call(rbind, agg_list)
  
  n_reps <- length(unique(x$rep))
  
  p <- ggplot2::ggplot(df, ggplot2::aes(x = k, y = loss, color = type)) +
    ggplot2::geom_line(data = df_mean, ggplot2::aes(x = k, y = mean_loss, color = type),
                       linewidth = 1) +
    ggplot2::geom_point(size = point_size, alpha = if (n_reps > 1) 0.6 else 1) +
    ggplot2::scale_x_continuous(breaks = sort(unique(x$k))) +
    ggplot2::labs(
      title = "NMF Cross-Validation",
      subtitle = if (n_reps > 1) paste0(n_reps, " replicates per rank") else NULL,
      x = "Rank (k)",
      y = "Loss",
      color = "Set"
    ) +
    ggplot2::theme_classic() +
    ggplot2::theme(legend.position = "bottom")
  
  # Mark optimal rank (minimum mean test loss)
  test_mean <- df_mean[df_mean$type == "Test", ]
  best_k <- test_mean$k[which.min(test_mean$mean_loss)]
  best_loss <- min(test_mean$mean_loss)
  
  p <- p + ggplot2::geom_vline(xintercept = best_k, linetype = "dashed",
                                color = "grey40", alpha = 0.7) +
    ggplot2::annotate("text", x = best_k, y = max(df$loss, na.rm = TRUE),
                      label = paste0("k=", best_k), vjust = -0.5,
                      color = "grey40", size = 3.5)
  
  if (interactive && requireNamespace("plotly", quietly = TRUE)) {
    p <- plotly::ggplotly(p)
  }
  
  return(p)
}
