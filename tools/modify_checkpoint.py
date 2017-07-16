import sys, getopt
import tensorflow as tf
import numpy as np

'''
Script to modify checkpoint files. Based on the gist: https://gist.github.com/batzner/7c24802dd9c5e15870b4b56e22135c96
'''

usage_str = 'python tensorflow_rename_variables.py path/to/checkpt/ ' \
    ' rename source_name target_name addprefix source_name  prefix add varname varvalue vartype delete varname overwrite'


def rename(checkpoint, renamevar, newvars, deletevar, overwrite):

    with tf.Session() as sess:
        for var_name, _ in tf.contrib.framework.list_variables(checkpoint):

            # Load the variable
            if deletevar is not None:
                if deletevar == var_name:
                    print("Omitting (deleting) variable %s" % var_name)
                    continue

            var = tf.contrib.framework.load_variable(checkpoint, var_name)
            currname = var_name

            if renamevar is not None:
                if var_name == renamevar[0]:
                    # modify it
                    currname = renamevar[1]
                    print('Modify %s  => %s ' % (renamevar[0], currname))
            # else:
            #     print("Ignoring variable %s " % var_name)
            # Declare the variable
            var = tf.Variable(var, name=currname)

            # check if the new var already exists
            if newvars is not None:
                if var_name == newvars[0]:
                    print("Variable to be created: %s already exists in the graph" % var_name)
                    exit()

        # add new variables, if needed
        if newvars is not None:
            varname, init_val, dtype = newvars
            print('Creating new variable name, value, type: %s, %s, %s' % ( varname, init_val, dtype ) )
            init_val = eval(init_val)
            dtype = eval(dtype)
            var = tf.Variable(init_val, dtype,name=varname)


        if overwrite:
            # Save the variables
            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())
            print("Saving modified model to ", checkpoint)
            saver.save(sess, checkpoint)
        else:
            print("Will not overwrite model - simulation run")


def main(argv):

    checkpoint = argv[0]
    overwrite = False
    newvar = None
    deletevar = None
    renamevar = None

    try:
        if "modify" in argv:
            i = (argv.index("modify"))
            oldname = (argv[i + 1])
            newname = (argv[i + 2])
            renamevar = (oldname, newname)

        if "overwrite" in argv:
            overwrite = True
        if "add" in argv:
            i = (argv.index("add"))
            vname = argv[i + 1]
            vvalue = argv[i + 2]
            vtype = argv[i + 3]
            newvar = (vname, vvalue, vtype)
        if "delete" in argv:
            i = (argv.index("delete"))
            deletevar = argv[i+1]

    except Exception:
        print(usage_str)
        exit()


    # check consistency
    ops = [deletevar, renamevar[0] if renamevar is not None else None, newvar[0] if newvar is not None else None]
    ops = [ o for o in ops if o is not None]
    if len(ops) != len(set(ops)):
        s = set()
        duplicates = set(x for x in ops if x in s or s.add(x))
        print("Specified duplicate operations for %s" % str(duplicates))
        exit()
    rename(checkpoint, renamevar, newvar, deletevar, overwrite)


if __name__ == '__main__':

    main(sys.argv[1:])
