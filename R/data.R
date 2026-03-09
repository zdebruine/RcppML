# ============================================================================
# Dataset Documentation
# ============================================================================

#' PBMC 3k Single-Cell RNA-seq Dataset (StreamPress Compressed)
#'
#' Single-cell RNA-seq gene expression data from ~2,700 peripheral blood
#' mononuclear cells (PBMCs). Shipped as StreamPress-compressed raw bytes
#' to meet CRAN tarball size limits.
#'
#' @format A \code{raw} vector containing StreamPress (.spz) compressed bytes.
#'   To obtain the sparse matrix, write the bytes to a temporary file and read
#'   with \code{\link{st_read}}:
#'   \preformatted{
#'   data(pbmc3k)
#'   tmp <- tempfile(fileext = ".spz")
#'   writeBin(pbmc3k, tmp)
#'   counts <- st_read(tmp)
#'   # counts is a dgCMatrix: 13,714 genes x 2,700 cells
#'   }
#'
#' @details
#' The underlying matrix is a \code{dgCMatrix} with 13,714 rows (genes) and
#' 2,700 columns (cells), containing 2,282,976 non-zero entries (integer UMI counts).
#' The data was compressed with StreamPress at fp32 precision, which is lossless
#' for integer count data.
#'
#' This dataset is commonly used for benchmarking single-cell analysis workflows.
#'
#' @source
#' 10x Genomics PBMC 3k dataset, filtered and processed.
#'
#' @examples
#' \donttest{
#' # Load the compressed bytes
#' data(pbmc3k)
#'
#' # Decompress to sparse matrix
#' tmp <- tempfile(fileext = ".spz")
#' writeBin(pbmc3k, tmp)
#' counts <- st_read(tmp)
#' dim(counts)  # 13714 x 2700
#' }
#'
#' @keywords datasets
"pbmc3k"

#' Acute Myelogenous Leukemia (AML) Dataset
#'
#' ATAC-seq chromatin accessibility data from 123 acute myelogenous leukemia (AML)
#' patient samples and 5 healthy hematopoietic stem/progenitor cell (HSPC) samples.
#' The dataset contains chromatin accessibility measurements across 824 genomic regions.
#'
#' @format A \code{matrix} (dense) with 824 rows (genomic regions) and
#'   135 columns (samples). Sample metadata is stored as an attribute.
#'   The data is stored as dense because it has very low sparsity (~0.2\% zeros).
#'
#'   Access metadata via \code{attr(aml, "metadata_h")}, which contains:
#'   \describe{
#'     \item{samples}{Character vector of sample IDs}
#'     \item{category}{Character vector indicating sample type (AML subtype or HSPC)}
#'   }
#'
#' @details
#' This dataset contains ATAC-seq chromatin accessibility data for studying acute
#' myelogenous leukemia. Samples represent different AML subtypes including:
#' \itemize{
#'   \item Common myeloid progenitors (CMP)
#'   \item Granulocyte-monocyte progenitors (GMP)
#'   \item Lymphoid-primed multi-potent progenitors (LMPP)
#'   \item Megakaryocyte-erythrocyte progenitors (MEP)
#'   \item Multi-potent progenitors (MPP)
#'   \item Healthy hematopoietic stem/progenitor cells (HSPC)
#' }
#'
#' @source
#' Corces et al. (2016). "Lineage-specific and single-cell chromatin accessibility
#' charts human hematopoiesis and leukemia evolution." Nature Genetics 48(10): 1193-1203.
#'
#' @examples
#' \donttest{
#' # Load dataset
#' data(aml)
#'
#' # Inspect structure
#' dim(aml)
#' table(attr(aml, "metadata_h")$category)
#'
#' # Run NMF (transpose so samples are columns)
#' result <- nmf(t(aml), k = 6, maxit = 100)
#' }
#'
#' @keywords datasets
"aml"

#' Hawaii Bird Species Frequency Dataset
#'
#' Frequency of bird species observations across survey grids in the Hawaiian islands.
#' This dataset contains presence/frequency data for 183 bird species across 1,183
#' spatial grid cells, along with metadata about survey locations and species characteristics.
#'
#' @format A \code{dgCMatrix} sparse matrix (183 x 1,183) with bird species
#'   frequency observations. Rows are bird species and columns are survey grid cells.
#'   Metadata is stored as attributes.
#'
#'   Access metadata via:
#'   \itemize{
#'     \item \code{attr(hawaiibirds, "metadata_h")} - Survey grid information (1,183 rows):
#'       \describe{
#'         \item{grid}{Character vector of grid cell identifiers}
#'         \item{island}{Factor indicating which Hawaiian island}
#'         \item{lat}{Numeric latitude coordinate}
#'         \item{lng}{Numeric longitude coordinate}
#'       }
#'     \item \code{attr(hawaiibirds, "metadata_w")} - Species information (183 rows):
#'       \describe{
#'         \item{species}{Character vector of species common names}
#'         \item{status}{Factor: "introduced" or "native"}
#'         \item{type}{Factor indicating bird type (e.g., "birds of prey", "seabirds")}
#'       }
#'   }
#'
#' @details
#' This dataset contains bird observation data from the Hawaiian Islands, useful for
#' ecological and biogeographic studies. The data is sparse, with many grid cells
#' containing no observations for most species.
#'
#' Bird types include:
#' \itemize{
#'   \item Birds of prey
#'   \item Coastal and wetland birds
#'   \item Land birds
#'   \item Seabirds
#'   \item Waterbirds
#' }
#'
#' @source
#' Data originally from RcppML package, derived from Hawaii bird observation surveys.
#'
#' @examples
#' \donttest{
#' # Load dataset
#' data(hawaiibirds)
#'
#' # Inspect structure
#' dim(hawaiibirds)
#' table(attr(hawaiibirds, "metadata_w")$status)
#' table(attr(hawaiibirds, "metadata_h")$island)
#'
#' # Run NMF to identify species co-occurrence patterns
#' result <- nmf(hawaiibirds, k = 10, maxit = 50)
#' }
#'
#' @keywords datasets
"hawaiibirds"

#' MovieLens Dataset
#'
#' Movie ratings data from the MovieLens 100K dataset, containing user ratings
#' for movies along with genre information.
#'
#' @format A \code{dgCMatrix} sparse matrix (3,867 x 610) containing movie ratings.
#'   Rows are movies, columns are users, and values are ratings from 1-5 stars.
#'   Most entries are zero (unrated). Genre information is stored as an attribute.
#'
#'   Access genre data via \code{attr(movielens, "genres")}, which returns an
#'   \code{ngCMatrix} sparse binary matrix (19 x 3,867) indicating genre membership.
#'   Rows are genres, columns are movies, and entries are TRUE/FALSE for genre membership.
#'
#' @details
#' This is a subset of the MovieLens dataset, widely used for demonstrating
#' collaborative filtering and recommendation systems. The data is highly sparse,
#' as most users have only rated a small fraction of available movies.
#'
#' The 19 genres include:
#' \itemize{
#'   \item Action, Adventure, Animation, Children
#'   \item Comedy, Crime, Documentary, Drama
#'   \item Fantasy, Film-Noir, Horror, Musical
#'   \item Mystery, Romance, Sci-Fi, Thriller
#'   \item War, Western, IMAX
#' }
#'
#' @source
#' MovieLens 100K Dataset from GroupLens Research.
#' \url{https://grouplens.org/datasets/movielens/}
#'
#' @examples
#' \donttest{
#' # Load dataset
#' data(movielens)
#'
#' # Inspect structure
#' dim(movielens)
#' dim(attr(movielens, "genres"))
#'
#' # Sparsity
#' Matrix::nnzero(movielens) / prod(dim(movielens))
#'
#' # Run NMF for collaborative filtering
#' result <- nmf(movielens, k = 20, maxit = 50)
#' }
#'
#' @keywords datasets
"movielens"

#' Olivetti Faces Dataset
#'
#' Grayscale face images from AT&T Laboratories Cambridge. This dataset contains
#' 400 face images (64x64 pixels) from 40 subjects, with 10 images per subject
#' showing different poses, expressions, and lighting conditions.
#'
#' @format A \code{dgCMatrix} sparse matrix (400 x 4096) containing grayscale
#'   face images. Each row is a flattened 64x64 pixel image with values in [0,1].
#'   Subject labels are stored as an attribute.
#'
#'   Access metadata via:
#'   \itemize{
#'     \item \code{attr(olivetti, "subject")} - Factor indicating which of 40 subjects
#'     \item \code{attr(olivetti, "image_shape")} - c(64, 64) image dimensions
#'     \item \code{attr(olivetti, "n_subjects")} - Number of subjects (40)
#'     \item \code{attr(olivetti, "images_per_subject")} - Images per subject (10)
#'   }
#'
#' @details
#' This dataset is commonly used for face recognition, clustering, and dimensionality
#' reduction benchmarks. The images show variation in:
#' \itemize{
#'   \item Facial expression (smiling, neutral, etc.)
#'   \item Head pose (left, right, up, down)
#'   \item Lighting conditions
#'   \item Accessories (glasses on/off in some cases)
#' }
#'
#' The true rank for NMF is 40 (number of subjects), though lower ranks may
#' capture common facial features and higher ranks may distinguish expression
#' and pose variations within subjects.
#'
#' To reshape a row back to an image:
#' \code{matrix(olivetti[i,], nrow=64, ncol=64, byrow=TRUE)}
#'
#' @source
#' AT&T Laboratories Cambridge (formerly Olivetti Research Laboratory).
#' \url{https://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html}
#'
#' Original images 92x112, downsampled to 64x64 in sklearn version.
#'
#' @examples
#' \donttest{
#' # Load dataset
#' data(olivetti)
#'
#' # Inspect
#' dim(olivetti)  # 400 x 4096
#' table(attr(olivetti, "subject"))  # 10 images per subject
#'
#' # Visualize first face
#' face_img <- matrix(olivetti[1,], nrow=64, ncol=64, byrow=TRUE)
#' image(t(face_img)[,64:1], col=grey.colors(256))
#'
#' # Run NMF to discover face components (small k for speed)
#' model <- nmf(t(olivetti), k = 5, maxit = 10)
#' }
#'
#' @keywords datasets
"olivetti"

#' Golub ALL-AML Dataset (Brunet et al. 2004)
#'
#' Gene expression data from the landmark Golub et al. (1999) study, consisting
#' of bone marrow samples from acute lymphoblastic leukemia (ALL) and acute
#' myeloid leukemia (AML) patients. This is the 5000 most variable genes subset
#' used in the seminal NMF paper by Brunet et al. (2004, PNAS).
#'
#' @format A \code{dgCMatrix} sparse matrix (38 x 5000) containing gene expression
#'   values. Rows are patient samples, columns are genes. Cancer type labels are
#'   stored as an attribute.
#'
#'   Access metadata via:
#'   \itemize{
#'     \item \code{attr(golub, "cancer_type")} - Factor: "ALL" or "AML"
#'     \item \code{attr(golub, "n_all")} - Number of ALL samples (27)
#'     \item \code{attr(golub, "n_aml")} - Number of AML samples (11)
#'   }
#'
#' @details
#' This dataset is one of the most widely used benchmarks for class discovery
#' and cancer subtype identification. The original study identified gene expression
#' signatures that distinguish ALL from AML.
#'
#' NMF analysis reveals:
#' \itemize{
#'   \item \strong{True rank = 2}: The two cancer types (ALL vs AML)
#'   \item \strong{Optimal rank ~3}: Brunet et al. found rank 3 more informative,
#'         potentially capturing ALL subtypes (B-cell vs T-cell)
#' }
#'
#' This dataset has been preprocessed to include only the 5000 most variable genes
#' for computational efficiency, following Brunet et al. (2004).
#'
#' @source
#' Golub et al. (1999). "Molecular classification of cancer: class discovery and
#' class prediction by gene expression monitoring." Science 286(5439): 531-537.
#'
#' Brunet et al. (2004). "Metagenes and molecular pattern discovery using matrix
#' factorization." PNAS 101(12): 4164-4169.
#'
#' Data retrieved via NMF package (Gaujoux and Seoighe).
#'
#' @examples
#' \donttest{
#' # Load dataset
#' data(golub)
#'
#' # Inspect
#' dim(golub)  # 38 samples x 5000 genes
#' table(attr(golub, "cancer_type"))  # 27 ALL, 11 AML
#'
#' # Run NMF for class discovery
#' model <- nmf(t(golub), k = 2, maxit = 100)  # Transpose: genes x samples
#'
#' # Try rank 3 as suggested by Brunet et al.
#' model3 <- nmf(t(golub), k = 3, maxit = 100)
#' }
#'
#' @keywords datasets
"golub"

#' MNIST Full Digits Dataset
#'
#' Full MNIST handwritten digit dataset containing 70,000 grayscale images
#' of digits 0-9. Each image is 28x28 pixels flattened to 784 features.
#'
#' @format A \code{matrix} (dense) with pixel intensities.
#'   Rows are samples, columns are features.
#'
#' @details
#' This is the complete MNIST dataset, commonly used for benchmarking machine
#' learning algorithms. The data is stored as a dense matrix (not sparse) since
#' handwritten digit images have substantial non-zero content.
#'
#' For NMF, it's recommended to:
#' \itemize{
#'   \item Normalize pixel values to [0,1] by dividing by 255
#'   \item Transpose to have pixels as rows, samples as columns
#'   \item Use a subset for faster experimentation
#' }
#'
#' The true rank is 10 (number of digit classes), though higher ranks may capture
#' stroke variations and writing styles within digits.
#'
#' @source
#' MNIST database of handwritten digits. Yann LeCun, Corinna Cortes, Christopher J.C. Burges.
#' \url{http://yann.lecun.com/exdb/mnist/}
#'
#' @examples
#' \donttest{
#' data(digits_full)
#' dim(digits_full)
#' model <- nmf(digits_full, k = 10, maxit = 50)
#' }
#'
#' @keywords datasets
"digits_full"
