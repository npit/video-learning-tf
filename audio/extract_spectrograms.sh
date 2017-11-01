folder="/run/media/nik/SAMSUNG/Data/msc-thesis/UCF101/videos/"
outfolder="/run/media/nik/SAMSUNG/Data/msc-thesis/UCF101/audio/spectrograms_ts_1"
timeslice="1"
python2 DL_generate_specs.py -i "${folder}" -t "${timeslice}" -w -o "${outfolder}"
