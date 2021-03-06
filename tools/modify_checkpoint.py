import sys, getopt
import tensorflow as tf
import numpy as np
import os
import argparse
import shutil
import re

'''
Script to modify checkpoint files. Based on the gist: https://gist.github.com/batzner/7c24802dd9c5e15870b4b56e22135c96
'''



def rename(checkpoint, delete_list, rename_list, create_list, regex_list, outpath):

    
    with tf.Session() as sess:
        # cycle through variables
        variable_names = []
        variable_data = []
        numdel, numcre, numren, numreg = 0,0,0,0
        print("Reading checkpoint",checkpoint)
        for var_name, _ in tf.contrib.framework.list_variables(checkpoint):
            # Load the variable, if it's not to be deleted
            if var_name in delete_list:
                print("Delete [%s]." % var_name)
                numdel += 1
                continue

            var = tf.contrib.framework.load_variable(checkpoint, var_name)
            curr_name = var_name

            if var_name in rename_list:
                # modify it
                print('Rename [%s]  => [%s].' % (curr_name, rename_list[curr_name]))
                curr_name = rename_list[curr_name]
                numren += 1
            else:
                matches = [i for i in range(len(regex_list)) if re.match(regex_list[i][0], curr_name)]
                if matches:
                    if len(matches) > 1:
                        print("Variable name {} matches more than one input regex: {}".format(var_name, str([regex_list[i][0] for i in matches])))
                    idx = matches[0]
                    regexed_name = re.sub(regex_list[idx][0], regex_list[idx][1], curr_name)
                    print('Regex rename [%s]  => [%s] from regex [%s],[%s].' % (curr_name, regexed_name, regex_list[idx][0], regex_list[idx][1]))
                    curr_name = regexed_name
                    numreg +=1

            # Declare the variable in the session
            var = tf.Variable(var, name=curr_name)
            variable_names.append(curr_name)
            variable_data.append(var.shape)


        # add new variables, if not already there
        if create_list:
            create_conflict = [ x for x in [c[0] for c in create_list] if x in variable_names]
            if create_conflict:
                print("Variables to create already exist in the checkpoint!")
                exit(1)
            for newvar in create_list:
                varname, init_val, dtype = newvar
                print('Creating new variable name, value, type: %s, %s, %s' % ( varname, init_val, dtype ) )
                var = tf.Variable(init_val, dtype,name=varname)
                variable_names.append(varname)
                variable_data.append(var.shape)
                numcre +=1

        # print resulting graph
        print("Created %d, renamed %d, regexed %d, deleted %d." % (numcre, numren, numreg, numdel))
        print()
        print("Resulting graph:")
        for i,(name, data) in enumerate(zip(variable_names, variable_data)):
            print(i+1,":",name,data)
        print()

        
        # Save the variables, if any change occurred
        if outpath is not None and any([x > 0 for x in [numdel, numcre, numren, numreg]]):
            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())
            print("Saving modified model to", outpath)
            saver.save(sess, outpath)
            if outpath != checkpoint:
                # potentially copy associated files, if not overwriting existing
                additional_suffixes = [".snap"]
                for suff in additional_suffixes:
                    s_orig = checkpoint + suff
                    s_dest = outpath + suff
                    if suff != s_dest:
                        print("Saving %s suffix file to [%s]" % (suff, s_dest))
                        shutil.copyfile(s_orig, s_dest)
                    else:
                        print("%s suffix file [%s] already in destination" % (suff, s_dest))

        else:
            print("Will not overwrite model (simulation run)")





if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--operation",nargs="*",dest="operations", action="append")
    parser.add_mutually_exclusive_group()
    parser.add_argument("--write",action="store_true")
    parser.add_argument("--outpath")
    parser.add_argument("--overwrite",action="store_true")

    args = parser.parse_args()
    if not args.operations:
        print("No operation specified.")
        exit(1)
    operations = ["delete","rename","create","regex"]
    delete_list, rename_list, create_list, regex_list = [], {}, [], []
    for op in args.operations:
        if len(op) < 2:
            print("Operation needs at least one operand.")
            exit(1)
        operation, opargs = op[0],op[1:]
        if operation not in operations:
            print("Undefined operation:",operation)
            exit(1)
        if operation == "delete":
            if not opargs:
                print("No args for delete operation.")
                exit(1)
            delete_list.extend(opargs)
        elif operation == "rename":
            if len(opargs) % 2 != 0:
                print("Rename requires tuples for arguments.")
                exit(1)
            tuplelist = [tuple(opargs[x:x + 2]) for x in range(0, len(opargs), 2)]
            for (origname,newname) in tuplelist:
                if origname in rename_list:
                    print("Specified a rename operation for variable [%s] more than once. (new names: %s, %s)" % (origname, rename_list[origname], newname))
                    exit(1)
                rename_list[origname] = newname
        elif operation == "regex":
            if len(opargs) % 2 != 0:
                print("Regex requires tuples for arguments.")
                exit(1)
            tuplelist = [tuple(opargs[x:x + 2]) for x in range(0, len(opargs), 2)]
            for (origname,newname) in tuplelist:
                if origname in regex_list:
                    print("Specified a regex operation for variable [%s] more than once. (new names: %s, %s)" % (origname, regex_list[origname], newname))
                    exit(1)
                regex_list.append((origname, newname))
        elif operation == "create":
            name, value, type = tuple(opargs)
            create_list.append((name, eval(value), eval(type)))

    outpath = None
    if args.overwrite:
        outpath = args.checkpoint
    elif args.write:
        outpath = args.checkpoint + ".modified"
    elif args.outpath:
        outpath = args.outpath


    rename(args.checkpoint, delete_list, rename_list, create_list, regex_list, outpath)
