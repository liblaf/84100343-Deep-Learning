#!/bin/bash
set -o errexit
set -o nounset
set -o pipefail

readarray -t figs < <(fd --no-ignore --extension ".png" --exclude "*-crop*" . ./gm/fig/)
for fig in "${figs[@]}"; do
  magick "$fig" -crop 104x274+0+0 "${fig%.png}-crop.png"
done
