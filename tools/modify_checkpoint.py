import sys, getopt
import tensorflow as tf
import numpy as np

'''
Script to modify checkpoint files. Based on the gist: https://gist.github.com/batzner/7c24802dd9c5e15870b4b56e22135c96
'''

usage_str = 'python tensorflow_rename_variables.py path/to/checkpt/ ' \
    'layername  rename target_name addprefix prefix add varname varvalue vartype overwrite'


def rename(checkpoint, layername, newname, prefix, newvars, overwrite):

    with tf.Session() as sess:
        for var_name, _ in tf.contrib.framework.list_variables(checkpoint):
            # Load the variable
            var = tf.contrib.framework.load_variable(checkpoint, var_name)
            currname = var_name

            if var_name == layername:
                # modify it
                if newname is not None:
                    currname = newname

                if prefix is not None:
                    currname = prefix + currname

                # if new_name == var_name:
                #     continue

                if not overwrite:

                    print('Would modify %s  => %s ' % (layername, currname))
                else:
                    print('Modifying %s  => %s ' % (layername, currname))

            # Declare the variable
            var = tf.Variable(var, name=currname)

        # add new variables, if needed
        if newvars is not None:
            varname, init_val, dtype = newvars
            print('Creating new variable: %s, %s, %s' % (init_val, varname, dtype))
            init_val = eval(init_val)
            dtype = eval(dtype)
            var = tf.Variable(init_val, dtype,name=varname)


        if overwrite:
            # Save the variables
            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())
            print("Saving modified model to ", checkpoint)
            saver.save(sess, checkpoint)


def main(argv):

    checkpoint = argv[0]
    layername = argv[1]
    newname = None
    prefix = None
    overwrite = False
    newvars = None

    try:
        if "rename" in argv:
            i = (argv.index("rename"))
            newname = (argv[i+1])

        if "addprefix" in argv:
            i = (argv.index("addprefix"))
            prefix = argv[i+1]
        if "overwrite" in argv:
            overwrite = True
        if "add" in argv:
            i = (argv.index("add"))
            vname = argv[i + 1]
            vvalue = argv[i + 2]
            vtype = argv[i + 3]
            newvars = (vname, vvalue, vtype)

    except Exception:
        print(usage_str)
        exit()

    if layername is None:
        print("No layer name specified")

    rename(checkpoint, layername, newname, prefix, newvars, overwrite)


if __name__ == '__main__':

    main(sys.argv[1:])