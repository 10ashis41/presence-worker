#!/usr/bin/env bash
# Build the PREVIEW (client-safe) from the clean render.
#
#   ./watermark.sh final.mp4 preview.mp4 ["Custom Label"]
#
# The preview must be good enough to judge quality but useless to actually
# publish, so it is degraded on three independent axes — removing any one of
# them by hand still leaves the other two:
#   1. repeated diagonal watermark across the frame (not just a corner logo)
#   2. half resolution
#   3. a periodic audio duck, so the soundtrack can't be lifted cleanly
set -euo pipefail

IN="${1:?usage: watermark.sh <in.mp4> <out.mp4> [label]}"
OUT="${2:?usage: watermark.sh <in.mp4> <out.mp4> [label]}"
LABEL="${3:-PREVIEW · DO NOT USE}"
FONT="${FONT:-/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf}"

ffmpeg -y -loglevel error -i "$IN" \
  -vf "scale=iw/2:-2,\
drawtext=fontfile=${FONT}:text='${LABEL}':fontcolor=white@0.42:fontsize=w/18:x=(w-tw)/2:y=h*0.18,\
drawtext=fontfile=${FONT}:text='${LABEL}':fontcolor=white@0.42:fontsize=w/18:x=(w-tw)/2:y=h*0.48,\
drawtext=fontfile=${FONT}:text='${LABEL}':fontcolor=white@0.42:fontsize=w/18:x=(w-tw)/2:y=h*0.78,\
drawtext=fontfile=${FONT}:text='PREVIEW':fontcolor=white@0.16:fontsize=w/6:x=(w-tw)/2:y=(h-th)/2" \
  -af "volume='if(lt(mod(t,7),0.7),0.08,1)':eval=frame" \
  -c:v libx264 -preset veryfast -crf 30 -pix_fmt yuv420p \
  -c:a aac -b:a 96k -movflags +faststart \
  "$OUT"

echo "preview: $(du -h "$OUT" | cut -f1)"
