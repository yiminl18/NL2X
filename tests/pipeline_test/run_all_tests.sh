#!/bin/bash

# Script to run all Python test files in subdirectories

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Running all pipeline tests..."
echo "=============================="

# Find all .py files in subdirectories and run them
test_files=$(find . -mindepth 2 -maxdepth 2 -type f -name "*.py" | sort)

total_tests=0
passed_tests=0
failed_tests=0

for test_file in $test_files; do
    total_tests=$((total_tests + 1))
    echo ""
    echo "Running: $test_file"
    echo "------------------------------"

    if python3 "$test_file"; then
        passed_tests=$((passed_tests + 1))
        echo "✓ PASSED: $test_file"
    else
        failed_tests=$((failed_tests + 1))
        echo "✗ FAILED: $test_file"
    fi
done

echo ""
echo "=============================="
echo "Test Summary:"
echo "  Total:  $total_tests"
echo "  Passed: $passed_tests"
echo "  Failed: $failed_tests"
echo "=============================="

if [ $failed_tests -eq 0 ]; then
    exit 0
else
    exit 1
fi
