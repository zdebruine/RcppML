#!/bin/bash
# Fix Rcpp 1.1.0 compileAttributes() bug
# 
# Rcpp 1.1.0 generates a spurious 'RcppExport void info(/* FIXME */);'
# forward declaration and '{"info", (DL_FUNC) &info, -1}' registration entry
# in RcppExports.cpp. This causes an undefined symbol error at runtime.
#
# This appears to be triggered by the Rcpp_sp_info() function name.
# Run this script after every compileAttributes() or roxygenise() call.
#
# Usage: bash tools/fix_rcpp_info_bug.sh

EXPORTS="src/RcppExports.cpp"

if [ ! -f "$EXPORTS" ]; then
    echo "Error: $EXPORTS not found. Run from package root."
    exit 1
fi

# Count occurrences before fix (check for any spurious pattern)
DECL_COUNT=$(grep -c '^RcppExport void' "$EXPORTS" 2>/dev/null || true)
# Filter to only count FIXME declarations (the bug pattern)
FIXME_COUNT=$(grep -c 'RcppExport void.*FIXME' "$EXPORTS" 2>/dev/null || true)
REG_FIXME_COUNT=$(grep -c '(DL_FUNC).*-1}' "$EXPORTS" 2>/dev/null || true)

# Handle empty output
FIXME_COUNT=${FIXME_COUNT:-0}
REG_FIXME_COUNT=${REG_FIXME_COUNT:-0}

if [ "$FIXME_COUNT" -eq 0 ] && [ "$REG_FIXME_COUNT" -eq 0 ]; then
    echo "No info bug found in $EXPORTS — already clean."
    exit 0
fi

# Remove ALL spurious FIXME declarations and -1 registrations
sed -i '/^RcppExport void.*\/\* FIXME \*\//d' "$EXPORTS"
sed -i '/(DL_FUNC).*-1}/d' "$EXPORTS"

echo "Fixed Rcpp bug: removed $FIXME_COUNT FIXME declaration(s) and $REG_FIXME_COUNT spurious registration(s)."
