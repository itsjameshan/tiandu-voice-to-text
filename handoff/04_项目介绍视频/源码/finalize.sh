#!/usr/bin/env bash
# Concatenate the per-shot segments and mux the narration track.
set -euo pipefail
cd "$(dirname "$0")"
: > segs/list.txt
for i in $(seq -w 1 12); do
  echo "file 'seg${i}.mp4'" >> segs/list.txt
done
ffmpeg -y -loglevel error -f concat -safe 0 -i segs/list.txt -c copy video_only.mp4
DUR=$(ffprobe -v error -show_entries format=duration -of csv=p=0 video_only.mp4)
ffmpeg -y -loglevel error -i video_only.mp4 -i narration_mix.wav \
  -map 0:v:0 -map 1:a:0 -c:v copy \
  -af "loudnorm=I=-16:TP=-1.5:LRA=11,aresample=48000" -c:a aac -b:a 192k \
  -t "$DUR" -movflags +faststart final.mp4
ffprobe -v error -show_entries stream=codec_name,width,height,r_frame_rate,duration,nb_frames,sample_rate,channels \
  -show_entries format=duration,size -of default=nw=1 final.mp4
