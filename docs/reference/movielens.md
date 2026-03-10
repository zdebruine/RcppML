# MovieLens Dataset

Movie ratings data from the MovieLens 100K dataset, containing user
ratings for movies along with genre information.

## Usage

``` r
movielens
```

## Format

A `dgCMatrix` sparse matrix (3,867 x 610) containing movie ratings. Rows
are movies, columns are users, and values are ratings from 1-5 stars.
Most entries are zero (unrated). Genre information is stored as an
attribute.

Access genre data via `attr(movielens, "genres")`, which returns an
`ngCMatrix` sparse binary matrix (19 x 3,867) indicating genre
membership. Rows are genres, columns are movies, and entries are
TRUE/FALSE for genre membership.

## Source

MovieLens 100K Dataset from GroupLens Research.
<https://grouplens.org/datasets/movielens/>

## Details

This is a subset of the MovieLens dataset, widely used for demonstrating
collaborative filtering and recommendation systems. The data is highly
sparse, as most users have only rated a small fraction of available
movies.

The 19 genres include:

- Action, Adventure, Animation, Children

- Comedy, Crime, Documentary, Drama

- Fantasy, Film-Noir, Horror, Musical

- Mystery, Romance, Sci-Fi, Thriller

- War, Western, IMAX

## Examples

``` r
# \donttest{
# Load dataset
data(movielens)

# Inspect structure
dim(movielens)
#> [1] 3867  610
dim(attr(movielens, "genres"))
#> [1]   19 3867

# Sparsity
Matrix::nnzero(movielens) / prod(dim(movielens))
#> [1] 0.03189578

# Run NMF for collaborative filtering
result <- nmf(movielens, k = 20, maxit = 50)
# }
```
