#!/usr/bin/env bash
# script to limit to one item per instance. Eg, in files with entries
# path/to/image/1.jpg label1 label2
# path/to/image/1.jpg label3

# this will keep one occurence of the image

[ $# -lt 1 ] && echo "Give the  .paths.txt file." && exit

awk '!seen[$1]++' "$1"
