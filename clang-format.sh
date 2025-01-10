#!/bin/sh

# Get the number of logical CPU cores
num_cores=$(nproc)
# Find and format files in parallel using the number of logical CPU cores
find . -regex '.*\.\(cpp\|cc\|h\|c\|hpp\|tpp\)' | xargs -P "$num_cores" -n 1 clang-format -style=file -i

