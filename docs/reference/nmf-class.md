# nmf S4 Class

The S4 class for NMF model results.

## Slots

- `w`:

  feature factor matrix

- `d`:

  scaling diagonal vector

- `h`:

  sample factor matrix

- `misc`:

  list containing optional components including tol (tolerance), iter
  (iterations), loss (final loss value), loss_type (loss function used),
  runtime (in seconds), w_init (initial w matrix), test_mask (CV test
  set), test_seed (CV seed), test_fraction (CV holdout fraction),
  train_loss (CV training loss), test_loss (CV test loss), and best_iter
  (CV best iteration)

## See also

[`nmf`](https://zdebruine.github.io/RcppML/reference/nmf.md),
[`evaluate`](https://zdebruine.github.io/RcppML/reference/evaluate.md),
[`align`](https://zdebruine.github.io/RcppML/reference/align.md),
[`predict,nmf-method`](https://zdebruine.github.io/RcppML/reference/nmf-class-methods.md)
