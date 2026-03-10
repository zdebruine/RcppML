# Hawaii Bird Species Frequency Dataset

Frequency of bird species observations across survey grids in the
Hawaiian islands. This dataset contains presence/frequency data for 183
bird species across 1,183 spatial grid cells, along with metadata about
survey locations and species characteristics.

## Usage

``` r
hawaiibirds
```

## Format

A `dgCMatrix` sparse matrix (183 x 1,183) with bird species frequency
observations. Rows are bird species and columns are survey grid cells.
Metadata is stored as attributes.

Access metadata via:

- `attr(hawaiibirds, "metadata_h")` - Survey grid information (1,183
  rows):

  - grid:

    Character vector of grid cell identifiers

  - island:

    Factor indicating which Hawaiian island

  - lat:

    Numeric latitude coordinate

  - lng:

    Numeric longitude coordinate

- `attr(hawaiibirds, "metadata_w")` - Species information (183 rows):

  - species:

    Character vector of species common names

  - status:

    Factor: "introduced" or "native"

  - type:

    Factor indicating bird type (e.g., "birds of prey", "seabirds")

## Source

Data originally from RcppML package, derived from Hawaii bird
observation surveys.

## Details

This dataset contains bird observation data from the Hawaiian Islands,
useful for ecological and biogeographic studies. The data is sparse,
with many grid cells containing no observations for most species.

Bird types include:

- Birds of prey

- Coastal and wetland birds

- Land birds

- Seabirds

- Waterbirds

## Examples

``` r
# \donttest{
# Load dataset
data(hawaiibirds)

# Inspect structure
dim(hawaiibirds)
#> [1]  183 1183
table(attr(hawaiibirds, "metadata_w")$status)
#> 
#> introduced     native 
#>         40        143 
table(attr(hawaiibirds, "metadata_h")$island)
#> 
#>    Hawaii Kahoolawe     Kauai     Lanai      Maui   Molokai      Oahu    Puuwai 
#>       593         4       183         5       148        19       229         2 

# Run NMF to identify species co-occurrence patterns
result <- nmf(hawaiibirds, k = 10, maxit = 50)
# }
```
