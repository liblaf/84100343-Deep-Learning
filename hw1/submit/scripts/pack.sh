#!/bin/bash
set -o errexit
set -o nounset
set -o pipefail

tmpdir=$(mktemp --directory)
trap 'rm --force --recursive -- "$tmpdir"' EXIT
readarray -t files < <(git ls-files)
files+=(part1/README.pdf part2/README.pdf)
for file in "${files[@]}"; do
  mkdir --parents -- "$tmpdir/$(dirname -- "$file")"
  cp --archive -- "$file" "$tmpdir/$file"
done
ouch compress "$tmpdir"/* submit.zip
