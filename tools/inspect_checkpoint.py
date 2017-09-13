
import sys


from tensorflow.python import pywrap_tensorflow


def print_tensors_in_checkpoint_file(file_name, tensor_name, print_values):
  """Prints tensors in a checkpoint file.

  If no `tensor_name` is provided, prints the tensor names and shapes
  in the checkpoint file.

  If `tensor_name` is provided, prints the content of the tensor.

  Args:
    file_name: Name of the checkpoint file.
    tensor_name: Name of the tensor in the checkpoint file to print.
    all_tensors: Boolean indicating whether to print all tensors.
  """
  all_tensors = True if not tensor_name else False
  count = 0
  try:
    reader = pywrap_tensorflow.NewCheckpointReader(file_name)
    if all_tensors:
      var_to_shape_map = reader.get_variable_to_shape_map()
      for key in var_to_shape_map:
        print("tensor_name: ", key)
        if print_values:
            print("tensor_value: ",reader.get_tensor(key))
        count = count  + 1
    elif not tensor_name:
      print(reader.debug_string().decode("utf-8"))
    else:
      print("tensor_name: ", tensor_name)
      if print_values:
        print("tensor_value: ",reader.get_tensor(tensor_name))
      count = count + 1
  except Exception as e:  # pylint: disable=broad-except
    print(str(e))
    if "corrupted compressed block contents" in str(e):
      print("It's likely that your checkpoint file has been compressed "
            "with SNAPPY.")
    if ("Data loss" in str(e) and
        (any([e in file_name for e in [".index", ".meta", ".data"]]))):
      proposed_file = ".".join(file_name.split(".")[0:-1])
      v2_file_error_template = """
It's likely that this is a V2 checkpoint and you need to provide the filename
*prefix*.  Try removing the '.' and extension.  Try:
inspect checkpoint --file_name = {}"""
      print(v2_file_error_template.format(proposed_file))
  print("%d tensors" % count)

args = sys.argv[1:]

def usage():
    print("Usage: %s path/to/checkpoint  e|tensor=tensorname  e|print=True|False" % sys.argv[0])
    exit()

if not args:
    usage()


checkpoint_path = args[0]
print_values = False
filter_tensor = ""
for arg in args[1:]:
    try:
        if arg.startswith("tensor="):
            filter_tensor = arg.split("=")[1]
        elif arg.startswith("print="):
            # print values
            tensorname=arg.split("=")[1]
            print_values = eval(tensorname)
        else:
          raise Exception("Undefined arg [%s]" % arg)
    except Exception:
        usage()


print("Checkpoint file %s, printing values: %s" % (checkpoint_path, str(print_values)))
print_tensors_in_checkpoint_file(file_name=checkpoint_path, tensor_name=filter_tensor, print_values = print_values)
