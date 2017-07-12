#!/usr/bin/env bash                                                                                                                                                          
[ $# -lt 1 ] && echo "Gotta provide a config file." && exit 1
configfile="$1"
errorfile="error_${configfile}_$(date  +'%Y%m%d_%H:%M:%S')"
echo "Using error file ${errorfile}"
python3 run_task.py "${configfile}"  2>"${errorfile}"
