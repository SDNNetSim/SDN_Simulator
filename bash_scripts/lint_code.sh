#!/bin/bash

set -e  # Fail on any error

# Navigate to repo root
SCRIPT_DIR=$(dirname "$0")
REPO_DIR=$(realpath "$SCRIPT_DIR/..")
cd "$REPO_DIR"

# Debugging: Log current directory
echo "Current working directory: $(pwd)"

# Check pylint installation
if ! command -v pylint &> /dev/null; then
  echo "pylint not found. Install it with 'pip install pylint'."
  exit 1
fi

# Identify Python files, excluding specific files or directories
echo "Identifying Python files..."

# Add paths to ignore using -path and -prune (ONLY ONCE).
PYTHON_FILES=$(find . \( \
  -path ./bash\* -o \
  -path ./.venv \
\) -prune -o -name "*.py" -print)

# Debugging: Log all found Python files
echo "Found Python files:"
echo "$PYTHON_FILES"

# Run pylint on each file
echo "Running pylint..."
for file in $PYTHON_FILES; do
  echo "Linting $file"
  pylint "$file" || exit 1
done

echo "Linting completed successfully!"
