import sys, os, yaml
import subprocess

"""
Function to run and evaluate the K last checkpoints of a given run.
"""

def wrap_quotes(arg):
    if " " in arg:
        return '"' + arg + '"'
    return arg

# number of checkpoints to evaluate
num_checkpoints = 5

if len(sys.argv) < 2:
    print("Give config file.")
    exit(1)

# parse checkpoints
#####################
config_file = sys.argv[1]
with open(config_file,"r") as f:
    config = yaml.load(f)['run']

resume_file = config['resume_file']
run_folder = config['run_folder']

if not resume_file == "latest":
    print("Resume file has to be 'latest' for multi-checkpoint evaluation")
    exit(1)

checkpoints = []
with open(os.path.join(run_folder,"checkpoints","checkpoint"),"r") as f:
    for line in f:
        chkp = line.strip().split(maxsplit=1)[1:]
        if len(chkp) > 1:
            print("Parsed non-unit element from checkpoint file:", chkp)
            exit(1)
        chkp = chkp[0]
        if chkp.startswith('"') or chkp.startswith("'"):
            chkp = chkp[1:-1]
        checkpoints.append(chkp)

# skip first line
checkpoints = checkpoints[1:]
if len(checkpoints) != num_checkpoints:
    print("Limiting number to %d checkpoints available" % len(checkpoints))
print("Checkpoints:")
num_checkpoints = min(num_checkpoints, len(checkpoints))
checkpoints = checkpoints[-num_checkpoints:]

for line in checkpoints:
    print(line)

# write configuration files
###########################
config_files = []
for i in range(len(checkpoints)):
    conffile = config_file + ".%d" % (i+1)
    config_files.append(conffile)
    curr_config = config
    value = wrap_quotes(checkpoints[i])
    curr_config['resume_file'] = value
    curr_config['phase'] = "defs.phase.val"
    curr_config = { "run" : curr_config }
    with open(config_files[-1],"w") as f:
        yaml.dump(curr_config, f, default_flow_style = False)

# run each validation run
for conf in config_files:
    cmd = ("python3 run_task.py " + wrap_quotes(conf)).split(maxsplit=2)
    print(cmd)
    #subprocess.run(cmd)
