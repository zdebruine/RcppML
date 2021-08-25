test_that("Testing nnls", {
  
  # a system of equations where 'a' is a matrix and 'b' is either a vector or a matrix
  a <- matrix(c(5.905428, 5.085229, 4.762723, 4.777141, 3.675643,
              5.085229, 7.430083, 6.225316, 6.513046, 4.180541,
              4.762723, 6.225316, 7.091737, 6.086963, 4.102543,
              4.777141, 6.513046, 6.086963, 7.234511, 4.227595,
              3.675643, 4.180541, 4.102543, 4.227595, 5.235357), 
              ncol = 5)
  b <- matrix(c(5.671745, 7.153883, 6.440488, 6.211075, 4.557587))
  b_matrix <- matrix(c(0.42802924, 0.46490343, 0.06213912, 0.43574293, 
                0.34564226, 0.76865630, 0.81245858, 0.80472978,
                0.26059196, 0.59442416, 0.16599049, 0.87390953,
                0.67386522,  0.74391732, 0.34301505), nrow = 5)
  true_nnls_solution <- c(0.26332579, 0.61837006, 0.14175903, 0.00000000, 0.08079708) # results from multiway::fnnls(a, b)

  # nnls solution should be equal to true solution
  expect_equal(true_nnls_solution, as.vector(nnls(a, b, cd_maxit = 1000, cd_tol = 1e-9)), tolerance = 1e-6)

  # solution from a parallelized matrix "b" should be the same as the unparallelized vector "b"
  expect_equal(as.vector(nnls(a, matrix(b_matrix[,2]))), nnls(a, b_matrix)[,2], tolerance = 1e-5)

    # check that incompatible sizes give an error
  expect_error(nnls(a, matrix(1:3)));
})