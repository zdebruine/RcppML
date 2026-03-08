#!/usr/bin/env python3
"""Add @examples to st_* functions in R/streampress.R that lack them."""
import re

with open("R/streampress.R", "r") as f:
    content = f.read()

# Map function name -> example block to insert before @export
examples = {
    "st_read_obs": """#' @examples
#' \\dontrun{
#' obs <- st_read_obs("data.spz")
#' head(obs)
#' }
#'""",
    "st_read_var": """#' @examples
#' \\dontrun{
#' var <- st_read_var("data.spz")
#' head(var)
#' }
#'""",
    "st_slice_cols": """#' @examples
#' \\dontrun{
#' mat <- st_slice_cols("data.spz", cols = 1:10)
#' dim(mat)
#' }
#'""",
    "st_slice_rows": """#' @examples
#' \\dontrun{
#' mat <- st_slice_rows("data.spz", rows = 1:100)
#' dim(mat)
#' }
#'""",
    "st_slice": """#' @examples
#' \\dontrun{
#' mat <- st_slice("data.spz", rows = 1:100, cols = 1:10)
#' dim(mat)
#' }
#'""",
    "st_chunk_ranges": """#' @examples
#' \\dontrun{
#' ranges <- st_chunk_ranges("data.spz")
#' print(ranges)
#' }
#'""",
    "st_map_chunks": """#' @examples
#' \\dontrun{
#' # Compute column sums per chunk
#' st_map_chunks("data.spz", function(chunk, s, e) Matrix::colSums(chunk))
#' }
#'""",
    "st_obs_indices": """#' @examples
#' \\dontrun{
#' idx <- st_obs_indices("data.spz", cell_type == "B cell")
#' length(idx)
#' }
#'""",
    "st_filter_rows": """#' @examples
#' \\dontrun{
#' mat <- st_filter_rows("data.spz", cell_type == "B cell")
#' dim(mat)
#' }
#'""",
    "st_filter_cols": """#' @examples
#' \\dontrun{
#' mat <- st_filter_cols("data.spz", highly_variable == TRUE)
#' dim(mat)
#' }
#'""",
    "st_write_list": """#' @examples
#' \\dontrun{
#' mats <- list(mat1, mat2)
#' st_write_list(mats, "combined.spz")
#' }
#'""",
}

for func_name, example_block in examples.items():
    # Use string replacement instead of regex to avoid backslash issues
    # Look for @export\n<func_name> and insert example before @export
    export_line = f"#' @export\n{func_name} <-"
    if export_line in content:
        content = content.replace(export_line, f"{example_block}\n{export_line}", 1)
        print(f"Added example to {func_name}")
    else:
        print(f"WARNING: Could not find insertion point for {func_name}")

with open("R/streampress.R", "w") as f:
    f.write(content)

print("Done!")
