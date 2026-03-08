#' Simulate Swimmer Dataset
#' 
#' @description Generate a synthetic "swimmer" dataset consisting of stick figure images
#' in various swimming poses. This is a classic benchmark dataset for NMF, originally
#' described in Donoho and Stodden (2003). Each image is a 32x32 pixel representation of
#' a stick figure with a torso and four limbs (left arm, right arm, left leg, right leg),
#' where each limb can be in one of 4 positions.
#'
#' @param n_images number of images to generate (default: 256 for all combinations)
#' @param style either "stick" for stick figures or "gaussian" for smoothed Gaussian blobs (default: "stick")
#' @param sigma standard deviation for Gaussian smoothing when style = "gaussian" (default: 1.5)
#' @param noise standard deviation of Gaussian noise to add (default: 0)
#' @param seed seed for random number generation
#' @param return_factors logical, if TRUE return the true W and H factors (default: FALSE)
#' 
#' @details
#' The swimmer dataset consists of stick figure images with:
#' \itemize{
#'   \item A fixed torso (circle for head, vertical line for body)
#'   \item 4 limbs, each with 4 possible angles: 0, 45, 90, 135 degrees from body
#'   \item Total of 16 "limb factors" (4 limbs x 4 positions each)
#'   \item 256 possible combinations when sampling all positions
#' }
#' 
#' The true rank of this dataset is 16 (the number of unique limb positions).
#' When \code{return_factors = TRUE}, the function returns the true generative
#' factors W (256 x 16) and H (16 x 1024), where each column of H represents
#' one limb position pattern.
#' 
#' Style options:
#' \itemize{
#'   \item \code{style = "stick"}: Sharp lines for limbs (binary image)
#'   \item \code{style = "gaussian"}: Smoothed with Gaussian kernel for softer edges
#' }
#'
#' @return If \code{return_factors = FALSE}, returns a sparse matrix (n_images x 1024)
#'   where each row is a flattened 32x32 image. If \code{return_factors = TRUE}, returns
#'   a list with components:
#'   \describe{
#'     \item{A}{The data matrix (n_images x 1024)}
#'     \item{w}{True factor matrix W (n_images x 16)}
#'     \item{h}{True factor matrix H (16 x 1024)}
#'     \item{limb_positions}{Matrix (n_images x 4) indicating limb positions}
#'   }
#'
#' @seealso \code{\link{simulateNMF}}, \code{\link{nmf}}
#' @export
#' @importFrom stats rnorm
#' 
#' @examples
#' \donttest{
#' # Generate all 256 swimmer combinations
#' swimmers <- simulateSwimmer()
#' dim(swimmers)  # 256 x 1024
#' 
#' # Generate random subset with Gaussian smoothing
#' swimmers_smooth <- simulateSwimmer(n_images = 100, style = "gaussian", seed = 123)
#' 
#' # Get true factors for validation
#' swimmer_data <- simulateSwimmer(return_factors = TRUE)
#' # Run NMF and compare to true factors
#' model <- nmf(swimmer_data$A, k = 16, maxit = 100)
#' }
#'
#' @references
#' Donoho, D. and Stodden, V. (2003). "When does non-negative matrix factorization 
#' give a correct decomposition into parts?" Advances in Neural Information Processing 
#' Systems 16.
#'
simulateSwimmer <- function(n_images = 256, style = c("stick", "gaussian"), 
                            sigma = 1.5, noise = 0, seed = NULL, 
                            return_factors = FALSE) {
  
  if (!is.null(seed)) set.seed(seed)
  style <- match.arg(style)
  
  img_size <- 32
  n_pixels <- img_size * img_size
  
  # Create coordinate grids
  x <- matrix(rep(1:img_size, img_size), nrow = img_size, byrow = TRUE)
  y <- matrix(rep(1:img_size, times = img_size), nrow = img_size)
  
  # Center coordinates
  cx <- img_size / 2
  cy <- img_size / 2
  
  # Helper: Gaussian blob centered at (x0, y0) with spread sigma
  gaussian_blob <- function(x0, y0, sig = 2) {
    exp(-((x - x0)^2 + (y - y0)^2) / (2 * sig^2))
  }
  
  # Helper: Line from (x0,y0) to (x1,y1) as Gaussian tube
  gaussian_line <- function(x0, y0, x1, y1, width = 1.5) {
    # Distance from each pixel to line segment
    dx <- x1 - x0
    dy <- y1 - y0
    len_sq <- dx^2 + dy^2
    
    if (len_sq < 1e-6) {
      return(gaussian_blob(x0, y0, width))
    }
    
    # Project each point onto line
    t <- pmax(0, pmin(1, ((x - x0) * dx + (y - y0) * dy) / len_sq))
    proj_x <- x0 + t * dx
    proj_y <- y0 + t * dy
    
    dist_sq <- (x - proj_x)^2 + (y - proj_y)^2
    exp(-dist_sq / (2 * width^2))
  }
  
  # Torso: fixed for all swimmers
  torso <- gaussian_blob(cx, cy - 5, sig = 2.5) + gaussian_line(cx, cy - 2, cx, cy + 5, width = 1.2)
  if (style == "stick") {
    torso <- (torso > 0.3) * 1.0
  }
  
  # Define limb positions: 4 limbs x 4 angles each = 16 factors
  # Angles: 0°(down/neutral), 45°, 90°(horizontal), 135°
  angles <- c(0, 45, 90, 135) * pi / 180
  limb_length <- 8
  
  # Limb attachment points and base directions
  limbs <- list(
    left_arm  = list(x0 = cx - 1, y0 = cy, base_angle = pi),      # extends left
    right_arm = list(x0 = cx + 1, y0 = cy, base_angle = 0),       # extends right
    left_leg  = list(x0 = cx - 1, y0 = cy + 5, base_angle = pi),  # extends left-down
    right_leg = list(x0 = cx + 1, y0 = cy + 5, base_angle = 0)    # extends right-down
  )
  
  # Generate 16 limb position images (the H matrix rows)
  n_factors <- 16
  limb_images <- vector("list", n_factors)
  factor_idx <- 1
  
  for (limb_name in names(limbs)) {
    limb <- limbs[[limb_name]]
    
    for (angle in angles) {
      # Calculate limb endpoint
      total_angle <- limb$base_angle + angle
      x1 <- limb$x0 + limb_length * cos(total_angle)
      y1 <- limb$y0 + limb_length * sin(total_angle)
      
      if (style == "gaussian") {
        img <- gaussian_line(limb$x0, limb$y0, x1, y1, width = 1.2)
      } else {
        # Stick: threshold the Gaussian line
        img <- (gaussian_line(limb$x0, limb$y0, x1, y1, width = 0.8) > 0.2) * 1.0
      }
      
      limb_images[[factor_idx]] <- img
      factor_idx <- factor_idx + 1
    }
  }
  
  # Normalize limb images
  for (i in 1:n_factors) {
    limb_images[[i]] <- limb_images[[i]] / max(limb_images[[i]])
  }
  torso <- torso / max(torso)
  
  # Determine which limb combinations to generate
  if (n_images >= 256) {
    # All 256 combinations (4^4)
    limb_combinations <- expand.grid(
      left_arm = 1:4,
      right_arm = 1:4,
      left_leg = 1:4,
      right_leg = 1:4
    )
    n_images <- 256
  } else {
    # Random sample
    limb_combinations <- data.frame(
      left_arm = sample(1:4, n_images, replace = TRUE),
      right_arm = sample(1:4, n_images, replace = TRUE),
      left_leg = sample(1:4, n_images, replace = TRUE),
      right_leg = sample(1:4, n_images, replace = TRUE)
    )
  }
  
  # Build matrices
  A <- matrix(0, n_images, n_pixels)
  W <- matrix(0, n_images, n_factors)
  H <- matrix(0, n_factors, n_pixels)
  
  # H matrix: each row is a flattened limb position image
  for (i in 1:n_factors) {
    H[i, ] <- as.vector(t(limb_images[[i]]))
  }
  
  # Generate each swimmer image
  for (i in 1:n_images) {
    img <- torso
    
    # Map limb choices to factor indices
    factor_indices <- c(
      limb_combinations$left_arm[i],                   # 1-4
      limb_combinations$right_arm[i] + 4,              # 5-8
      limb_combinations$left_leg[i] + 8,               # 9-12
      limb_combinations$right_leg[i] + 12              # 13-16
    )
    
    # Add limbs
    for (idx in factor_indices) {
      img <- img + limb_images[[idx]]
      W[i, idx] <- 1
    }
    
    # Flatten
    A[i, ] <- as.vector(t(img))
  }
  
  # Add noise
  if (noise > 0) {
    A <- A + matrix(rnorm(n_images * n_pixels, 0, noise), n_images, n_pixels)
    A[A < 0] <- 0
  }
  
  # Convert to sparse
  A <- .to_dgCMatrix(A)
  
  if (return_factors) {
    list(
      A = A,
      w = W,
      h = H,
      limb_positions = as.matrix(limb_combinations)
    )
  } else {
    A
  }
}
