#!/usr/bin/env python3
import os, sys, configparser
from pathlib import Path
from utils_ import  *

current_dir = os.path.abspath(os.path.curdir)
examples_dir = Path(current_dir).parent
root_dir = sys.argv[1] if len(sys.argv)>1 else Path(current_dir).parent.parent
config_file_name=sys.argv[2] if len(sys.argv) > 1 else "config.ini.activity_rec_example"
# get directories
print("Root directory : ", root_dir)
# work out run configuration
config = configparser.ConfigParser()
config.read(os.path.join(root_dir,"config.ini.example"))

config_run = config['run']
config_run['workflow'] ='defs.workflows.singleframe'
config_run['run_id'] = '\"activity_rec_example\"'
config_run['resume_file'] = 'None'
config_run['run_folder'] = '"' + current_dir  + '"'
config_run['input_mode'] = 'defs.input_mode.video'

# work out serialization configuration
config_serialize=config['serialize']
video_data_folder= os.path.join(examples_dir,"data","videos")
pathsfile = os.path.join(video_data_folder,"data.train")
os.system("cp '%s' '%s'" % (os.path.join(video_data_folder,"paths.txt"), pathsfile))
config_serialize['input_files']='["' + pathsfile + '"]'
config_serialize['num_frames_per_clip']='3'
config_serialize['clipframe_mode']='defs.clipframe_mode.rand_clips'

# write config
filepath = os.path.join(root_dir, config_file_name)
print("Writing example activity recognition configuration file to %s" % filepath)
with open(filepath,'w') as fp:
    config.write(fp)


