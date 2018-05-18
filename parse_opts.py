from utils_ import error
from ast import literal_eval as fromstr
from collections import Iterable


def parse_seq(arg):
    if type(arg) == list or type(arg) == tuple:
        return arg
    try:
        return fromstr(arg)
    except:
        error("Unable to literal-eval expression [%s]" % arg)

def to_list(arg):
    if type(arg) in [list, tuple]:
        return arg
    return [arg]
