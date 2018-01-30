import sys, yaml
from os.path import join, basename
import os
import subprocess

"""
Function to run and evaluate the K last checkpoints of a given run.
"""

# number of checkpoints to evaluate. Note that tensorflow has a similar param, keeping at most k checkpoints.
# set to -1 to evaluate all available
num_checkpoints = -1

if len(sys.argv) < 2:
    print("Give config file.")
    exit(1)

opts = sys.argv[2:]

# parse checkpoints
#####################
config_file = sys.argv[1]
with open(config_file,"r") as f:
    config = yaml.load(f)['run']

resume_file = config['resume_file']
run_folder = config['run_folder']

checkpoints = []
with open(join(run_folder,"checkpoints","checkpoint"),"r") as f:
    for line in f:
        chkp = line.strip().split(maxsplit=1)[1:]
        if len(chkp) > 1:
            print("Parsed non-unit element from checkpoint file:", chkp)
            exit(1)
        chkp = chkp[0]
        if chkp.startswith('"') or chkp.startswith("'"):
            chkp = chkp[1:-1]
        checkpoints.append(chkp)

# skip first line on the 'checkpoints' file
checkpoints = checkpoints[1:]
if len(checkpoints) != num_checkpoints and num_checkpoints > 1:
    print("Limiting number to %d checkpoints available" % len(checkpoints))
    num_checkpoints = min(num_checkpoints, len(checkpoints))
    checkpoints = checkpoints[-num_checkpoints:]
print("Checkpoints:")
for line in checkpoints:
    print(line)

# write configuration files
###########################
config_files, run_ids = [], []
base_run_id = config["rund_id"] if "run_id" in config else ""
for i in range(len(checkpoints)):
    conffile = config_file + ".%d" % (i+1)
    config_files.append(conffile)
    config['resume_file'] = checkpoints[i]
    config['phase'] = "defs.phase.val"
    config['run_id'] = base_run_id + "multiple_eval_%d" % (i+1)
    run_ids.append(config['run_id'])
    curr_config = { "run" : config }
    if not "onlyprint" in opts:
        with open(config_files[-1],"w") as f:
            yaml.dump(curr_config, f, default_flow_style = False)

# run each validation run
if not "onlyprint" in opts:
    for i, conf in enumerate(config_files):
        cmd = ("python3 run_task.py " + conf).split(maxsplit=2)
        print("Running %d/%d validation, with command:" % (i+1, len(config_files)),cmd)
        subprocess.run(cmd)

# print out accuracies
print("Getting results from",run_folder)
dirfiles = [ff for ff in os.listdir(run_folder)]
for i,(conf, rid) in enumerate(zip(config_files, run_ids)):
    accfiles = [f for f in dirfiles if rid in  f and "accuracy" in f]
    if len(accfiles) > 1:
        print("(!) Multiple accuracy files for",rid,":",accfiles)
    if not accfiles:
        print("(!) No accuracy files for",rid,":",accfiles)
        continue
    accfile = accfiles[0]
    with open(join(run_folder,accfile),"r") as f:
        accuracy = f.read()
    print(rid,accfile,basename(checkpoints[i]),accuracy)
