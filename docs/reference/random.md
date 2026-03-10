# Random distributions and samples

`r_sample` is just like
[`base::sample`](https://rdrr.io/r/base/sample.html), only faster.
`r_sample` takes a sample of the specified size from the elements of `x`
using replacement if indicated.

These functions generate random distributions (uniform, normal, or
binomial) just like their base R counterparts (`runif`, `rnorm`, and
`rbinom`), but faster.

## Usage

``` r
r_sample(x, size = NULL, replace = FALSE)

r_unif(n, min = 0, max = 1)

r_binom(n, size = 1, inv_prob = 2)
```

## Arguments

- x:

  either a positive integer giving the number of items to choose from,
  or a vector of elements to shuffle or from which to choose. See
  'Details'.

- size:

  number of trials (one or more)

- replace:

  should sampling be with replacement?

- n:

  number of observations

- min:

  finite lower limit of the uniform distribution

- max:

  finite upper limit of the uniform distribution

- inv_prob:

  inverse probability of success for each trial, must be integral (e.g.
  50 percent success = 2, 10 percent success = 10)

## Value

For `r_sample`: a vector of sampled elements from `x`.

For `r_unif`: a numeric vector of random values from the uniform
distribution.

For `r_binom`: a numeric vector of random values from the binomial
distribution.

## Details

All RNGs make use of Marsaglia's xorshift method to generate random
integers.

`r_unif` takes the random integer and divides it by the seed and returns
the floating decimal portion of the result.

## See also

[`r_matrix`](https://zdebruine.github.io/RcppML/reference/r_matrix.md),
[`r_sparsematrix`](https://zdebruine.github.io/RcppML/reference/r_matrix.md)

## Examples

``` r
if (FALSE) { # \dontrun{
# draw all integers from 1 to 10 in a random order
r_sample(10)

# shuffle a vector of values
v <- r_unif(3)
v
v_ <- r_sample(v)
v_

# draw values from a vector
r_sample(r_unif(100), 3)

# draw some integers between 1 and 1000
r_sample(1000, 3)
} # }
if (FALSE) { # \dontrun{
# simulate a uniform distribution
v <- r_unif(10000)
plot(density(v))

# simulate a binomial distribution
v <- r_binom(10000, 100, inv_prob = 10)
hist(v)
sum(v) / length(v)
# ~10 because 100 trials at 10 percent success odds
#   is about 10 successes per element

# get successful trials in a bernoulli distribution
v <- r_binom(100, 1, 20)
successful_trials <- slot(as(v, "nsparseVector"), "i")
successful_trials
} # }
```
