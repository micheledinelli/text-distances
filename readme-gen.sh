#!/bin/bash

# Step 1: Convert Jupyter Notebooks to Markdown
jupyter nbconvert --to markdown --no-input text-distance-analysis.ipynb --output text-distance-analysis.md
jupyter nbconvert --to markdown --no-inpu text-representation-analysis.ipynb --output text-representation-analysis.md 

# Step 2: Combine Markdown files into README.md
cat text-distance-analysis.md text-representation-analysis.md > README.md

# Step 3: Optionally, remove intermediate Markdown files
rm text-distance-analysis.md text-representation-analysis.md

echo "[![Run nbconvert](https://github.com/micheledinelli/text-distances/actions/workflows/readme.yaml/badge.svg?branch=main)](https://github.com/micheledinelli/text-distances/actions/workflows/readme.yaml)" >> README.md
