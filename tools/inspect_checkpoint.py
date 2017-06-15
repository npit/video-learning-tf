import tensorflow as tf
import sys
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

checkpoint_path = sys.argv[1]
print("Checkpoint file %s" % checkpoint_path)
print_tensors_in_checkpoint_file(file_name=checkpoint_path, tensor_name='',all_tensors=True) 
