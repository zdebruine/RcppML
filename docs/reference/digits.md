# MNIST Digits Dataset

MNIST handwritten digit dataset containing grayscale images of digits
0-9. Each image is 28x28 pixels flattened to 784 features.

## Usage

``` r
digits
```

## Format

A `matrix` (dense) with pixel intensities. Rows are samples, columns are
features.

## Source

MNIST database of handwritten digits. Yann LeCun, Corinna Cortes,
Christopher J.C. Burges. <http://yann.lecun.com/exdb/mnist/>

## Details

This is the MNIST dataset, commonly used for benchmarking machine
learning algorithms. The data is stored as a dense matrix (not sparse)
since handwritten digit images have substantial non-zero content.

For NMF, it's recommended to:

- Normalize pixel values to `[0,1]` by dividing by 255

- Transpose to have pixels as rows, samples as columns

- Use a subset for faster experimentation

The true rank is 10 (number of digit classes), though higher ranks may
capture stroke variations and writing styles within digits.

## Examples

``` r
# \donttest{
data(digits)
dim(digits)
#> [1] 1797   64
model <- nmf(digits, k = 10, maxit = 50)
# }
```
