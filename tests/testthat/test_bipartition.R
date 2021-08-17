test_that("Testing RcppML::nmf", {
  
  A <- rbind(c(1.7009878, 0.00000000, 0.9542116, 0.68738015, 1.1690738012, 1.4190147, 0.8697167, 0.72520486, 0.00000000, 0.0000000),
             c(1.5901502, 0.61081441, 0.6841591, 0.00000000, 0.9110474099, 0.5809928, 0.0000000, 0.99028990, 0.01800186, 0.3915861),
             c(1.1629335, 2.13608402, 2.3780564, 2.71694620, 1.3074841130, 1.7009005, 0.0000000, 0.01130724, 1.59745476, 0.0000000),
             c(0.0000000, 0.00000000, 0.4145758, 0.01870036, 0.0000000000, 0.0000000, 0.0000000, 0.00000000, 0.00000000, 0.0000000),
             c(0.0000000, 0.90947564, 0.6896592, 0.00000000, 1.5718641979, 0.2359384, 0.9782686, 3.03143394, 0.00000000, 0.6474887),
             c(1.3965336, 0.00000000, 0.0000000, 0.00000000, 1.0002093844, 0.7561340, 0.0000000, 0.00000000, 0.89331994, 1.1115118),
             c(0.6743350, 0.08068094, 0.0000000, 1.40130937, 1.4421851828, 1.1075104, 0.1342477, 3.17786538, 2.56604624, 2.8281075),
             c(0.4790339, 0.89483168, 0.0000000, 0.33670458, 0.7380427447, 0.0000000, 2.9913736, 1.87699437, 2.86377032, 1.9204602),
             c(0.0000000, 0.00000000, 0.0000000, 1.30777912, 0.0000000000, 0.5205217, 2.2561151, 3.71541181, 1.19395162, 3.0003078),
             c(0.0000000, 0.00000000, 0.0000000, 0.00000000, 0.0001228284, 0.0000000, 3.3094624, 2.97985598, 2.85893710, 2.6126768))
  
  samples1 <- 0:5
  samples2 <- 6:9
  size1 <- 4
  size2 <- 6
  center1 <- rowMeans(A[,samples1 + 1])
  center2 <- rowMeans(A[,samples2 + 1])
  
  # test that bipartition is as expected (DENSE)
  model <- bipartition(A)
  expect_equal(model$dist, -1)
  expect_equal(model$size1 == 4 || model$size1 == 6, TRUE)
  if(model$size1 == 6){
    expect_equal(model$size2 == 4, TRUE)
    expect_equal(abs(sum(model$center1 - center1)) < 1e-5, TRUE)
    expect_equal(abs(sum(model$center2 - center2)) < 1e-5, TRUE)
  } else {
    expect_equal(model$size2 == 6, TRUE)
    expect_equal(abs(sum(model$center1 - center2)) < 1e-5, TRUE)
    expect_equal(abs(sum(model$center2 - center1)) < 1e-5, TRUE)
  }

  # test that bipartition is as expected (SPARSE)
  A <- as(A, "dgCMatrix")
  model <- bipartition(A)
  expect_equal(model$dist, -1)
  expect_equal(model$size1 == 4 || model$size1 == 6, TRUE)
  if(model$size1 == 6){
    expect_equal(model$size2 == 4, TRUE)
    expect_equal(abs(sum(model$center1 - center1)) < 1e-5, TRUE)
    expect_equal(abs(sum(model$center2 - center2)) < 1e-5, TRUE)
  } else {
    expect_equal(model$size2 == 6, TRUE)
    expect_equal(abs(sum(model$center1 - center2)) < 1e-5, TRUE)
    expect_equal(abs(sum(model$center2 - center1)) < 1e-5, TRUE)
  }
})