#!/usr/bin/env bash
[ $# -lt 1 ] && echo "Usage $(basename $0) <log-directory>" && exit 1
logdir="$1"
tensorboard --logdir="$logdir"
