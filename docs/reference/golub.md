# Golub ALL-AML Dataset (Brunet et al. 2004)

Gene expression data from the landmark Golub et al. (1999) study,
consisting of bone marrow samples from acute lymphoblastic leukemia
(ALL) and acute myeloid leukemia (AML) patients. This is the 5000 most
variable genes subset used in the seminal NMF paper by Brunet et al.
(2004, PNAS).

## Usage

``` r
golub
```

## Format

A `dgCMatrix` sparse matrix (38 x 5000) containing gene expression
values. Rows are patient samples, columns are genes. Cancer type labels
are stored as an attribute.

Access metadata via:

- `attr(golub, "cancer_type")` - Factor: "ALL" or "AML"

- `attr(golub, "n_all")` - Number of ALL samples (27)

- `attr(golub, "n_aml")` - Number of AML samples (11)

## Source

Golub et al. (1999). "Molecular classification of cancer: class
discovery and class prediction by gene expression monitoring." Science
286(5439): 531-537.

Brunet et al. (2004). "Metagenes and molecular pattern discovery using
matrix factorization." PNAS 101(12): 4164-4169.

Data retrieved via NMF package (Gaujoux and Seoighe).

## Details

This dataset is one of the most widely used benchmarks for class
discovery and cancer subtype identification. The original study
identified gene expression signatures that distinguish ALL from AML.

NMF analysis reveals:

- **True rank = 2**: The two cancer types (ALL vs AML)

- **Optimal rank ~3**: Brunet et al. found rank 3 more informative,
  potentially capturing ALL subtypes (B-cell vs T-cell)

This dataset has been preprocessed to include only the 5000 most
variable genes for computational efficiency, following Brunet et al.
(2004).

## Examples

``` r
# \donttest{
# Load dataset
data(golub)

# Inspect
dim(golub)  # 38 samples x 5000 genes
#> [1]   38 5000
table(attr(golub, "cancer_type"))  # 27 ALL, 11 AML
#> 
#> ALL AML 
#>  27  11 

# Run NMF for class discovery
model <- nmf(t(golub), k = 2, maxit = 100)  # Transpose: genes x samples

# Try rank 3 as suggested by Brunet et al.
model3 <- nmf(t(golub), k = 3, maxit = 100)
# }
```
