#!/usr/bin/env bash
cd ../..
rootdir="$(pwd)"
examples_dir="${rootdir}/examples"
curdir="${examples_dir}/activity_recognition"
export PYTHONPATH="$rootdir"

cd "${curdir}"
config_file_name="config.ini.activity_rec_example"
python3 prepare_activity_recognition.py "$rootdir" "$config_file_name"
cd "$rootdir"

# serialize data
python3 serialize.py "$config_file_name"
# move data to run folder
mv "${examples_dir}/data/videos/data.train"* "${curdir}/"
# run the task
python3 run_task.py "$config_file_name"

