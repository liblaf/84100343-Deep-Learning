#!/bin/bash
set -o errexit
set -o nounset
set -o pipefail

xhs --output "data_and_checkpoints.zip" --download "https://cloud.tsinghua.edu.cn/f/dc0ff3454fff44c9b626/?dl=1"
tmp_dir=$(mktemp -d)
trap 'command rm --force --recursive $tmp_dir' EXIT
ouch decompress --dir "$tmp_dir" --yes --hidden "data_and_checkpoints.zip"
mv "$tmp_dir/data_and_checkpoints/checkpoints" "checkpoints"
mv "$tmp_dir/data_and_checkpoints/data" "data"
