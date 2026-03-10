# Simulate Swimmer Dataset

Generate a synthetic "swimmer" dataset consisting of stick figure images
in various swimming poses. This is a classic benchmark dataset for NMF,
originally described in Donoho and Stodden (2003). Each image is a 32x32
pixel representation of a stick figure with a torso and four limbs (left
arm, right arm, left leg, right leg), where each limb can be in one of 4
positions.

## Usage

``` r
simulateSwimmer(
  n_images = 256,
  style = c("stick", "gaussian"),
  sigma = 1.5,
  noise = 0,
  seed = NULL,
  return_factors = FALSE
)
```

## Arguments

- n_images:

  number of images to generate (default: 256 for all combinations)

- style:

  either "stick" for stick figures or "gaussian" for smoothed Gaussian
  blobs (default: "stick")

- sigma:

  standard deviation for Gaussian smoothing when style = "gaussian"
  (default: 1.5)

- noise:

  standard deviation of Gaussian noise to add (default: 0)

- seed:

  seed for random number generation

- return_factors:

  logical, if TRUE return the true W and H factors (default: FALSE)

## Value

If `return_factors = FALSE`, returns a sparse matrix (n_images x 1024)
where each row is a flattened 32x32 image. If `return_factors = TRUE`,
returns a list with components:

- A:

  The data matrix (n_images x 1024)

- w:

  True factor matrix W (n_images x 16)

- h:

  True factor matrix H (16 x 1024)

- limb_positions:

  Matrix (n_images x 4) indicating limb positions

## Details

The swimmer dataset consists of stick figure images with:

- A fixed torso (circle for head, vertical line for body)

- 4 limbs, each with 4 possible angles: 0, 45, 90, 135 degrees from body

- Total of 16 "limb factors" (4 limbs x 4 positions each)

- 256 possible combinations when sampling all positions

The true rank of this dataset is 16 (the number of unique limb
positions). When `return_factors = TRUE`, the function returns the true
generative factors W (256 x 16) and H (16 x 1024), where each column of
H represents one limb position pattern.

Style options:

- `style = "stick"`: Sharp lines for limbs (binary image)

- `style = "gaussian"`: Smoothed with Gaussian kernel for softer edges

## References

Donoho, D. and Stodden, V. (2003). "When does non-negative matrix
factorization give a correct decomposition into parts?" Advances in
Neural Information Processing Systems 16.

## See also

[`simulateNMF`](https://zdebruine.github.io/RcppML/reference/simulateNMF.md),
[`nmf`](https://zdebruine.github.io/RcppML/reference/nmf.md)

## Examples

``` r
# \donttest{
# Generate all 256 swimmer combinations
swimmers <- simulateSwimmer()
dim(swimmers)  # 256 x 1024
#> [1]  256 1024

# Generate random subset with Gaussian smoothing
swimmers_smooth <- simulateSwimmer(n_images = 100, style = "gaussian", seed = 123)

# Get true factors for validation
swimmer_data <- simulateSwimmer(return_factors = TRUE)
# Run NMF and compare to true factors
model <- nmf(swimmer_data$A, k = 16, maxit = 100)
# }
```
