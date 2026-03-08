# Makefile for standalone C++ unit tests (no R required, only Eigen)
#
# Usage:
#   module load r/4.5.2
#   make            # Build all tests
#   make run        # Build and run
#   make clean      # Clean build artifacts
#
# Or via R (for approval-free execution):
#   Rscript -e 'system("cd tests/cpp && make run")'

# Auto-detect include paths from R
EIGEN_INCLUDE := $(shell Rscript -e 'cat(system.file("include", package="RcppEigen"))')
RCPP_INCLUDE  := $(shell Rscript -e 'cat(system.file("include", package="Rcpp"))')
R_INCLUDE     := $(shell Rscript -e 'cat(R.home("include"))')
PROJECT_INCLUDE := ../../inst/include

CXX      := g++
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra -Wno-unused-parameter \
            -I$(PROJECT_INCLUDE) \
            -I$(EIGEN_INCLUDE) \
            -I$(RCPP_INCLUDE) \
            -I$(R_INCLUDE) \
            -I. \
            -fopenmp -DEIGEN_DONT_PARALLELIZE

LDFLAGS  := -lgomp

# Test sources (each file has TEST_CASE definitions)
TEST_SRCS := test_rng.cpp test_config.cpp test_nnls.cpp test_gram.cpp test_nmf.cpp
TEST_OBJS := $(TEST_SRCS:.cpp=.o)

TARGET := run_tests

.PHONY: all run clean

all: $(TARGET)

$(TARGET): main.cpp $(TEST_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cpp test_framework.hpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET) $(TEST_OBJS)
