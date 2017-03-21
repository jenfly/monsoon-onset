#!/bin/bash -v

cd figs/anim

ffmpeg -r 30 -i frame%03d.png -vcodec mpeg4 -r 30 anim.mpeg

#ffmpeg -r 20 -f image2 -i myImage%04d.png -f mp4 -q:v 0 -vcodec mpeg4 -r 20 myVideo.mp4
