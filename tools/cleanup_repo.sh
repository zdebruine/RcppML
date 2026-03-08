#!/bin/bash
# RcppML Repository Cleanup Script
# Removes build artifacts, SLURM logs, and regenerable files
# Run from repo root: bash tools/cleanup_repo.sh

set -e

# Get repo root
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "=== RcppML Package Cleanup ==="
echo "Working directory: $REPO_ROOT"
echo ""

# Function to safely delete with confirmation
safe_delete() {
    local path="$1"
    local desc="$2"
    
    if [ -e "$path" ]; then
        echo "Deleting: $desc ($path)"
        rm -rf "$path"
    else
        echo "Not found (skipping): $desc ($path)"
    fi
}

# Function to delete matching files
delete_pattern() {
    local pattern="$1"
    local desc="$2"
    
    echo "Deleting: $desc (pattern: $pattern)"
    find . -name "$pattern" -type f -delete 2>/dev/null || true
}

echo "--- 1. Temporary Build Artifacts ---"
safe_delete "RcppML.Rcheck" "R CMD check output directory"
safe_delete "nohup.out" "nohup background process log"
safe_delete "inst/lib/RcppML_gpu.so" "Compiled GPU library"
delete_pattern "*.o" "Compiled object files"
delete_pattern "*.so" "Shared object libraries (src/)"
echo ""

echo "--- 2. SLURM Job Outputs ---"
delete_pattern "*.out" "SLURM stdout files"
delete_pattern "*.err" "SLURM stderr files"
# Keep only non-SLURM log files (fix_rcpp_info_bug logs are useful)
find tests/ benchmarks/logs/ tools/ -name "*_[0-9][0-9][0-9][0-9][0-9][0-9].log" -delete 2>/dev/null || true
echo ""

echo "--- 3. Compiled Benchmark Binaries ---"
# Delete executables without extensions in benchmarks/comprehensive/
for binary in profile_nmf nnls_bench gpu_production_bench gpu_production_bench_90 \
              gpu_production_bench_test hybrid_spmm_bench rank_selection_bench \
              spmm_vs_gemm_bench syrk_bench validate_adaptive debug_cublas \
              debug_full debug_full_dual adaptive_rhs_bench; do
    safe_delete "benchmarks/comprehensive/$binary" "Benchmark binary: $binary"
done
safe_delete "benchmarks/cuda/production_benchmark" "Legacy CUDA benchmark binary"
echo ""

echo "--- 4. Benchmark Results (Regenerable) ---"
safe_delete "benchmarks/results/*.csv" "Benchmark CSV files"
safe_delete "benchmarks/results/*.rds" "Benchmark RDS snapshots"
safe_delete "benchmarks/results/*.txt" "Benchmark text logs"
safe_delete "benchmarks/results/experiment_log.txt" "Experiment log"
safe_delete "benchmarks/results/overfitting_log.txt" "Overfitting log"
echo ""

echo "--- 5. Legacy CUDA Code ---"
safe_delete "benchmarks/cuda/legacy" "Legacy CUDA prototypes directory"
echo ""

echo "--- 6. Python Virtual Environment ---"
safe_delete ".venv" "Python virtual environment"
echo ""

echo "--- 7. Manuscript LaTeX Artifacts ---"
find manuscript/jss/ -name "*.aux" -o -name "*.log" -o -name "*.bbl" \
     -o -name "*.blg" -o -name "*.out" | xargs rm -f 2>/dev/null || true
safe_delete "manuscript/jss/missfont.log" "LaTeX missing font log"
echo ""

echo "--- 8. SparsePress Build Artifacts (OPTIONAL - commented by default) ---"
# Uncomment the line below if you want to delete SparsePress build directory
# safe_delete "sparsepress/build" "SparsePress build directory"
echo "Skipping sparsepress/build/ (uncomment in script to delete)"
echo ""

echo "=== Cleanup Summary ==="
echo ""
echo "Deleted:"
echo "  - R CMD check output (RcppML.Rcheck/)"
echo "  - SLURM job logs (*.out, *.err, job-specific *.log)"
echo "  - Compiled benchmark binaries (~15 executables)"
echo "  - Benchmark results (*.csv, *.rds, *.txt)"
echo "  - Legacy CUDA code (benchmarks/cuda/legacy/)"
echo "  - Python virtual environment (.venv/)"
echo "  - LaTeX artifacts (manuscript/jss/*.aux, *.log, etc.)"
echo ""
echo "Kept:"
echo "  - Source code (R/, src/, inst/include/)"
echo "  - Tests (tests/testthat/)"
echo "  - Documentation (man/, vignettes/)"
echo "  - Data (data/*.rda)"
echo "  - Benchmark scripts (benchmarks/R/, benchmarks/slurm/)"
echo "  - Manuscript source (manuscript/*.md, manuscript/code/)"
echo "  - Tools (tools/*.sh, tools/*.sbatch)"
echo ""

# Show new repository size
echo "=== Repository Size ==="
du -sh .
echo ""
echo "=== File Count by Type ==="
echo "R source files: $(find R/ -name "*.R" 2>/dev/null | wc -l)"
echo "C++ source files: $(find src/ inst/include/ -name "*.cpp" -o -name "*.hpp" -o -name "*.cu" 2>/dev/null | wc -l)"
echo "Test files: $(find tests/testthat/ -name "*.R" 2>/dev/null | wc -l)"
echo "Vignettes: $(find vignettes/ -name "*.Rmd" 2>/dev/null | wc -l)"
echo ""

echo "=== Next Steps ==="
echo "1. Review changes: git status"
echo "2. Test build: devtools::document(); devtools::check()"
echo "3. Run tests: devtools::test()"
echo "4. Check CRAN compliance: R CMD check --as-cran"
echo ""
echo "Cleanup complete!"
