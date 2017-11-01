folder="path/to/videos/"
outfolder="path/to/write/spectrogram/images"
timeslice="1"
python2 DL_generate_specs.py -i "${folder}" -t "${timeslice}" -w -o "${outfolder}"
