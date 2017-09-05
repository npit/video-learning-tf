#!/usr/bin/env bash
[ $# -lt 1 ] && echo "Usage $(basename $0) <log-directory>" && exit 1
logdir="$1"

active_tboard_pids="$(ps aux | grep tensorboard | grep -v grep | grep -v $0 | awk '{print $2}')"
if [ ! -z "$active_tboard_pids" ]; then
	echo "Active tensorboard instance(s) at pid(s) $active_tboard_pids"
	echo "Press enter to kill running instances and delete contents of $logdir ?"
	read -p ""

	rm $logdir/* -f
	kill $active_tboard_pids
fi

tensorboard --logdir="$logdir" & disown
