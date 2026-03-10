# Acute Myelogenous Leukemia (AML) Dataset

ATAC-seq chromatin accessibility data from 123 acute myelogenous
leukemia (AML) patient samples and 5 healthy hematopoietic
stem/progenitor cell (HSPC) samples. The dataset contains chromatin
accessibility measurements across 824 genomic regions.

## Usage

``` r
aml
```

## Format

A `matrix` (dense) with 824 rows (genomic regions) and 135 columns
(samples). Sample metadata is stored as an attribute. The data is stored
as dense because it has very low sparsity (~0.2% zeros).

Access metadata via `attr(aml, "metadata_h")`, which contains:

- samples:

  Character vector of sample IDs

- category:

  Character vector indicating sample type (AML subtype or HSPC)

## Source

Corces et al. (2016). "Lineage-specific and single-cell chromatin
accessibility charts human hematopoiesis and leukemia evolution." Nature
Genetics 48(10): 1193-1203.

## Details

This dataset contains ATAC-seq chromatin accessibility data for studying
acute myelogenous leukemia. Samples represent different AML subtypes
including:

- Common myeloid progenitors (CMP)

- Granulocyte-monocyte progenitors (GMP)

- Lymphoid-primed multi-potent progenitors (LMPP)

- Megakaryocyte-erythrocyte progenitors (MEP)

- Multi-potent progenitors (MPP)

- Healthy hematopoietic stem/progenitor cells (HSPC)

## Examples

``` r
# \donttest{
# Load dataset
data(aml)

# Inspect structure
dim(aml)
#> [1] 824 135
table(attr(aml, "metadata_h")$category)
#> 
#>       AML (GMP)     AML (L-MPP)       AML (MEP)   Control (GMP) Control (L-MPP) 
#>              93              12              15               5               5 
#>   Control (MEP) 
#>               5 

# Run NMF (transpose so samples are columns)
result <- nmf(t(aml), k = 6, maxit = 100)
# }
```
