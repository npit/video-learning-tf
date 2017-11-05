from os import makedirs, listdir, devnull
from os.path import exists, join, basename, isdir, abspath
import configparser
import argparse
import subprocess
"""
Script to run a single validation run for each input checkpoint.
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoints_folder', help='checkpoints folder.')
    parser.add_argument('run_folder', help='The validation run superfolder, which will in the end contain one run folder per checkpoint.')
    parser.add_argument('config_file', help='The validation configuration file to use')
    parser.add_argument('-checkpoint_key', required=False, default="resume_file",  help='The checkpoint key in the configuration file')
    parser.add_argument('-show_stdout', required=False, action='store_true', default=False, help='The checkpoint key in the configuration file')

    args = parser.parse_args()
    args.checkpoints_folder = abspath(args.checkpoints_folder)
    args.run_folder = abspath(args.run_folder)
    args.config_file = abspath(args.config_file)
    print("Evaluating checkpoints at", args.checkpoints_folder)
    print("On run folder", args.run_folder)
    print("Using config file", args.config_file)

    if not exists(args.run_folder):
        print("Creating the run folder.")
        makedirs(args.run_folder, exist_ok=True)
    else:
        if not isdir(args.run_folder):
            print("Run folder is not a directory.")
            exit(1)

    configp = configparser.ConfigParser()
    configp.read(args.config_file)

    ext=".index"
    checkpts = [ck[:-len(ext)] for ck in listdir(args.checkpoints_folder) if ck.endswith(ext)]

    for ridx, checkpoint_file in enumerate(checkpts):
        print()
        print("Using checkpoint [%s]" % checkpoint_file)
        checkpoint_file = join(args.checkpoints_folder, checkpoint_file)
        configp.set('run', args.checkpoint_key, '"%s"' % checkpoint_file)
        configp.set('run', 'do_training','False')
        configp.set('run', 'do_validation','True')
        # create run folder, write in config file
        current_run_folder = join(args.run_folder, "run_%d_%s" % (ridx+1, basename(checkpoint_file)))
        makedirs(current_run_folder, exist_ok=True)
        run_config_file = join(current_run_folder, basename(args.config_file))
        print("\tUsing conffile [%s]" % run_config_file)
        with open(run_config_file, "w") as fp: 
            configp.write(fp)

        # run it 
        print("Running task %d/%d at folder [%s]" % (1+ridx, len(checkpts), basename(current_run_folder)))
        print("==================================================")

        try:
            # run_task.main(run_config_file)
            if args.show_stdout:
                subprocess.run(["python3","run_task.py",run_config_file], check=True)
            else:
                with open(devnull,"w") as dnull:
                    subprocess.run(["python3", "run_task.py", run_config_file], check=True, stdout=dnull, stderr=subprocess.STDOUT)
        except Exception as ex:
            print(ex)
            exit(1)
        except Error as ex:
            print(ex)
            exit(1)
        print("==================================================")
        print("\tDone running task %d/%d at folder [%s]" % (1+ridx, len(checkpts), basename(current_run_folder)))






