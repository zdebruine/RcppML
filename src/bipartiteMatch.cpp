#include <Rcpp.h>
#include <RcppHungarian.h>

// [[Rcpp::export]]
Rcpp::List Rcpp_bipartite_match(Rcpp::NumericMatrix x) {

  std::vector<double> c(x.ncol());
  std::vector<std::vector<double>> cm(x.nrow(), c);
  for (int i = 0; i < x.nrow(); ++i) {
    for (int j = 0; j < x.ncol(); ++j)
      c[j] = x(i, j);
    cm[i] = c;
  }

  HungarianAlgorithm HungAlgo;
  std::vector<int> pairs;
  double cost = HungAlgo.Solve(cm, pairs);
  for(unsigned int i = 0; i < pairs.size(); ++i) ++pairs[i];
  
  return(Rcpp::List::create(
    Rcpp::Named("cost") = cost,
    Rcpp::Named("pairs") = pairs
  ));
}