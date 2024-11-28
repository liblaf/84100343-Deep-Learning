#!/bin/bash
set -o errexit -o nounset -o pipefail

git_root=$(git rev-parse --show-toplevel)
tmpdir=$(mktemp --directory)
trap 'rm --force --recursive -- "$tmpdir"' EXIT
readarray -t files < <(git ls-files)
files+=(
  part1-rnn/README.pdf
  part2-gnn/README.pdf
)
for file in "${files[@]}"; do
  mkdir --parents -- "$tmpdir/$(dirname -- "$file")"
  cp --archive -- "$file" "$tmpdir/$file"
done
cp --archive -- "$git_root/pixi.lock" "$tmpdir/pixi.lock"
cp --archive -- "$git_root/pyproject.toml" "$tmpdir/pyproject.toml"
ouch compress "$tmpdir"/* submit.zip
