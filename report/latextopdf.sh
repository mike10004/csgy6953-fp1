#!/bin/bash

TEX_FILE=$1
OUTPUT_DIR=$2

if [[ -z "$TEX_FILE" ]] ; then
#  echo "latextopdf.sh: usage: <TEX_FILE> [OUTPUT_DIR]" >&2
#  exit 1
   TEX_FILE=dlfp-report.tex
fi

if [[ -z "$OUTPUT_DIR" ]] ; then
  OUTPUT_DIR=$(dirname "$TEX_FILE")/output
fi

mkdir -p "${OUTPUT_DIR}"

exec latexmk -pdf -f -g -bibtex -deps -synctex=1 -interaction=nonstopmode -output-directory="${OUTPUT_DIR}" "${TEX_FILE}"
