import sys, yaml, argparse
from os.path import join, basename
import os
import subprocess
import re
from utils_ import prep_email, notify_email, get_run_checkpoints

"""
Function to run and evaluate the K last checkpoints of a given run.
"""

parser = argparse.ArgumentParser()
parser.add_argument("configfile")
parser.add_argument("--onlyprint", action="store_true")
# number of checkpoints to evaluate. Note that tensorflow has a similar param, keeping at most k checkpoints.
parser.add_argument("-num_checkpoints", type=int)
parser.add_argument("-omit_epochs", nargs="*", dest="omit")
args = parser.parse_args()


# parse checkpoints
#####################
with open(args.configfile,"r") as f:
    config = yaml.load(f)['run']

resume_file = config['resume_file']
run_folder = config['run_folder']
email_notify = config['logging']['email_notify']
if email_notify:
    sender, password, recipient = prep_email(email_notify)

if args.omit:
    args.omit = ["_ep_{}_".format(x) for x in args.omit]
raw_checkpoints = get_run_checkpoints(run_folder)
checkpoints = []
for chkp in raw_checkpoints:
    if chkp.startswith('"') or chkp.startswith("'"):
        chkp = chkp[1:-1]
    if args.omit and any([x in chkp for x in args.omit]):
        print("Omitting {} due to epoch restriction arguments.".format(chkp))
        continue
    checkpoints.append(chkp)

if args.num_checkpoints:
    if len(checkpoints) < args.num_checkpoints: print("Unable to run for {} checkpoints, as there are only {}".format(args.num_checkpoints, len(checkpoints)))
    if len(checkpoints) < args.num_checkpoints: print("Limiting evaluation from {} to the {} last checkpoints".format(len(checkpoints), args.num_checkpoints))
    num_checkpoints = min(args.num_checkpoints, len(checkpoints))
    checkpoints = checkpoints[-num_checkpoints:]
print("Checkpoints:")
for line in checkpoints:
    print(line)

# write configuration files
###########################
config_files, run_ids = [], []
base_run_id = config["rund_id"] if "run_id" in config else ""
for i in range(len(checkpoints)):
    checkpoint_path = checkpoints[i]
    conffile = os.path.splitext(args.configfile)[0] + "." + os.path.basename(checkpoint_path) + ".yml"
    config_files.append(conffile)
    config['resume_file'] = checkpoint_path
    config['phase'] = "defs.phase.val"
    config['run_id'] = base_run_id + "multiple_eval_%d" % (i+1)
    # no email notification
    config['logging']['email_notify'] = ""
    run_ids.append(config['run_id'])
    curr_config = { "run" : config }
    if not args.onlyprint:
        with open(config_files[-1],"w") as f:
            yaml.dump(curr_config, f, default_flow_style = False)

# run each validation run
if not args.onlyprint:
    for i, conf in enumerate(config_files):
        cmd = ("python3 run_task.py " + conf).split(maxsplit=2)
        print("Running %d/%d validation, with command:" % (i+1, len(config_files)),cmd)
        subprocess.run(cmd)
        
else:
    for i, conf in enumerate(config_files):
        print(conf)
    exit(1)

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

if email_notify:
    notify_email(sender, password, recipient, "Multi-chekcpoint evaluation complete.", msgtype="INFO")
